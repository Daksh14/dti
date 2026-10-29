//! The worker's only route to the network: `wasi:http` requests to the
//! coordinator and to the volunteer's data origins.
//!
//! Raw `wasi:sockets` stay disabled (the `WasiCtx` default), so this hook is
//! the single place outbound traffic is allowed, refused and counted. It also
//! enforces the data budget: successful responses from data origins stop once
//! their bodies add up to it.

use bytes::Bytes;
use http::Uri;
use http::uri::Scheme;
use http_body::{Body, Frame};
use http_body_util::BodyExt;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::task::{Context as TaskContext, Poll, ready};
use wasmtime::bail;
use wasmtime::error::Context;
use wasmtime_wasi_http::{Error, RequestOptions, WasiBody, WasiHttpHooks};

/// Scheme, host and port the worker may talk to. A host written `*.example.com`
/// matches every subdomain of `example.com`, but not `example.com` itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    https: bool,
    host: String,
    port: u16,
    subdomains: bool,
}

impl Origin {
    pub fn parse(url: &str) -> wasmtime::Result<Self> {
        let (plain, subdomains) = match url.split_once("://*.") {
            Some((scheme, rest)) => (format!("{scheme}://{rest}"), true),
            None => (url.to_string(), false),
        };
        let uri = plain
            .parse::<Uri>()
            .with_context(|| format!("invalid URL {url:?}"))?;
        let Some(origin) = Self::of(&uri) else {
            bail!("expected http:// or https:// with a host: {url:?}");
        };
        if subdomains && !uri.path().trim_end_matches('/').is_empty() {
            bail!("a wildcard origin cannot have a path: {url:?}");
        }
        Ok(Self {
            subdomains,
            ..origin
        })
    }

    fn of(uri: &Uri) -> Option<Self> {
        let https = match uri.scheme() {
            Some(s) if *s == Scheme::HTTPS => true,
            Some(s) if *s == Scheme::HTTP => false,
            _ => return None,
        };
        let authority = uri.authority()?;
        let host = authority.host().to_ascii_lowercase();
        if host.is_empty() {
            return None;
        }
        let port = authority.port_u16().unwrap_or(if https { 443 } else { 80 });
        Some(Self {
            https,
            host,
            port,
            subdomains: false,
        })
    }

    pub fn is_https(&self) -> bool {
        self.https
    }

    pub fn allows(&self, uri: &Uri) -> bool {
        Self::of(uri).is_some_and(|o| {
            o.https == self.https
                && o.port == self.port
                && if self.subdomains {
                    o.host
                        .strip_suffix(&self.host)
                        .is_some_and(|sub| sub.len() > 1 && sub.ends_with('.'))
                } else {
                    o.host == self.host
                }
        })
    }
}

#[derive(Default)]
pub struct NetStats {
    pub requests: AtomicU64,
    pub denied: AtomicU64,
    pub bytes_up: AtomicU64,
    pub bytes_down: AtomicU64,
    /// Bytes of successful responses from data origins; capped by the budget.
    pub data_down: AtomicU64,
}

type SendFuture = Box<
    dyn Future<
            Output = wasmtime_wasi_http::Result<(
                http::Response<WasiBody>,
                Box<dyn Future<Output = Result<(), Error>> + Send>,
            )>,
        > + Send,
>;

pub struct Egress {
    pub coordinator: Origin,
    pub data: Vec<Origin>,
    pub data_budget: u64,
    pub stats: Arc<NetStats>,
}

impl WasiHttpHooks for Egress {
    fn send_request(
        &mut self,
        request: http::Request<WasiBody>,
        options: Option<RequestOptions>,
        fut: Box<dyn Future<Output = Result<(), Error>> + Send>,
    ) -> SendFuture {
        _ = fut;
        let uri = request.uri();
        let is_data = if self.coordinator.allows(uri) {
            false
        } else if self.data.iter().any(|o| o.allows(uri)) {
            true
        } else {
            return self.deny(&request, "");
        };
        if is_data && self.stats.data_down.load(Relaxed) >= self.data_budget {
            return self.deny(&request, " (data budget used up)");
        }
        self.stats.requests.fetch_add(1, Relaxed);
        let stats = self.stats.clone();
        let budget = self.data_budget;
        let request = request.map(|body| counted(body, stats.clone(), |s| &s.bytes_up));
        Box::new(async move {
            let (response, io) = wasmtime_wasi_http::default_send_request(request, options).await?;
            let capped = is_data && response.status().is_success();
            let response = response.map(|body| {
                let body = counted(body, stats.clone(), |s| &s.bytes_down);
                if capped {
                    Capped {
                        inner: body,
                        stats,
                        budget,
                    }
                    .boxed_unsync()
                } else {
                    body
                }
            });
            Ok((response, Box::new(io) as Box<dyn Future<Output = _> + Send>))
        })
    }
}

impl Egress {
    fn deny(&self, request: &http::Request<WasiBody>, why: &str) -> SendFuture {
        self.stats.denied.fetch_add(1, Relaxed);
        println!(
            "[net]    DENIED {} {}{why}",
            request.method(),
            request.uri()
        );
        Box::new(async { Err(Error::HttpRequestDenied) })
    }
}

fn counted<B>(body: B, stats: Arc<NetStats>, counter: fn(&NetStats) -> &AtomicU64) -> WasiBody
where
    B: Body<Data = Bytes, Error = Error> + Send + 'static,
{
    body.map_frame(move |frame| {
        if let Some(data) = frame.data_ref() {
            counter(&stats).fetch_add(data.len() as u64, Relaxed);
        }
        frame
    })
    .boxed_unsync()
}

/// A data response body that fails once all data downloads together pass the
/// budget, whatever sizes the worker or the coordinator claimed.
struct Capped {
    inner: WasiBody,
    stats: Arc<NetStats>,
    budget: u64,
}

impl Body for Capped {
    type Data = Bytes;
    type Error = Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Error>>> {
        let frame = ready!(Pin::new(&mut self.inner).poll_frame(cx));
        if let Some(Ok(f)) = &frame
            && let Some(data) = f.data_ref()
        {
            let len = data.len() as u64;
            let total = self.stats.data_down.fetch_add(len, Relaxed) + len;
            if total > self.budget {
                println!("[net]    data budget used up; cutting off a download");
                return Poll::Ready(Some(Err(Error::HttpResponseBodySize(None))));
            }
        }
        Poll::Ready(frame)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allows(origin: &str, uri: &str) -> bool {
        Origin::parse(origin).unwrap().allows(&uri.parse().unwrap())
    }

    #[test]
    fn only_the_configured_origin_is_allowed() {
        let c = "https://coordinator.example.com";
        assert!(allows(c, "https://coordinator.example.com/v1/task/w1"));
        assert!(allows(c, "https://COORDINATOR.example.com:443/v1/join"));
        assert!(!allows(c, "http://coordinator.example.com/v1/join"));
        assert!(!allows(c, "https://coordinator.example.com:8443/"));
        assert!(!allows(c, "https://pool.example.net/"));
        assert!(!allows(c, "https://coordinator.example.com.evil.net/"));
        assert!(!allows(c, "https://sub.coordinator.example.com/"));
        assert!(!allows(c, "/relative/path"));

        let local = "http://127.0.0.1:8080";
        assert!(allows(local, "http://127.0.0.1:8080/v1/join"));
        assert!(!allows(local, "http://127.0.0.1/v1/join"));
        assert!(!allows(local, "http://localhost:8080/v1/join"));
    }

    #[test]
    fn wildcards_match_subdomains_only() {
        let cdn = "https://*.hf.co";
        assert!(allows(cdn, "https://us.aws.cdn.hf.co/xet-bridge-us/abc"));
        assert!(allows(cdn, "https://cdn-lfs.hf.co/x"));
        assert!(!allows(cdn, "https://hf.co/x"));
        assert!(!allows(cdn, "https://evilhf.co/x"));
        assert!(!allows(cdn, "https://cdn.hf.co.evil.net/x"));
        assert!(!allows(cdn, "http://cdn-lfs.hf.co/x"));
        assert!(!allows(cdn, "https://cdn-lfs.hf.co:8443/x"));
    }

    #[test]
    fn rejects_unusable_urls() {
        assert!(Origin::parse("ftp://example.com").is_err());
        assert!(Origin::parse("example.com").is_err());
        assert!(Origin::parse("https://*.hf.co/some/path").is_err());
    }
}
