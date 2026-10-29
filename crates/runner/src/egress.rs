//! The worker's only route to the network: `wasi:http` requests to one origin.
//!
//! Raw `wasi:sockets` stay disabled (the `WasiCtx` default), so this hook is
//! the single place outbound traffic is allowed, refused and counted.

use bytes::Bytes;
use http::Uri;
use http::uri::Scheme;
use http_body::Body;
use http_body_util::BodyExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use wasmtime::bail;
use wasmtime::error::Context;
use wasmtime_wasi_http::{Error, RequestOptions, WasiBody, WasiHttpHooks};

/// Scheme, host and port the worker may talk to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    https: bool,
    host: String,
    port: u16,
}

impl Origin {
    pub fn parse(url: &str) -> wasmtime::Result<Self> {
        let uri = url
            .parse::<Uri>()
            .with_context(|| format!("invalid coordinator URL {url:?}"))?;
        match Self::of(&uri) {
            Some(origin) => Ok(origin),
            None => bail!("coordinator URL must be http:// or https:// with a host: {url:?}"),
        }
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
        Some(Self { https, host, port })
    }

    pub fn is_https(&self) -> bool {
        self.https
    }

    pub fn allows(&self, uri: &Uri) -> bool {
        Self::of(uri).is_some_and(|o| o == *self)
    }
}

#[derive(Default)]
pub struct NetStats {
    pub requests: AtomicU64,
    pub denied: AtomicU64,
    pub bytes_up: AtomicU64,
    pub bytes_down: AtomicU64,
}

pub struct Egress {
    pub allowed: Origin,
    pub stats: Arc<NetStats>,
}

impl WasiHttpHooks for Egress {
    fn send_request(
        &mut self,
        request: http::Request<WasiBody>,
        options: Option<RequestOptions>,
        fut: Box<dyn Future<Output = Result<(), Error>> + Send>,
    ) -> Box<
        dyn Future<
                Output = wasmtime_wasi_http::Result<(
                    http::Response<WasiBody>,
                    Box<dyn Future<Output = Result<(), Error>> + Send>,
                )>,
            > + Send,
    > {
        _ = fut;
        if !self.allowed.allows(request.uri()) {
            self.stats.denied.fetch_add(1, Relaxed);
            println!("[net]    DENIED {} {}", request.method(), request.uri());
            return Box::new(async { Err(Error::HttpRequestDenied) });
        }
        self.stats.requests.fetch_add(1, Relaxed);
        let stats = self.stats.clone();
        let request = request.map(|body| counted(body, stats.clone(), |s| &s.bytes_up));
        Box::new(async move {
            let (response, io) = wasmtime_wasi_http::default_send_request(request, options).await?;
            let response = response.map(|body| counted(body, stats, |s| &s.bytes_down));
            Ok((response, Box::new(io) as Box<dyn Future<Output = _> + Send>))
        })
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
        assert!(!allows(c, "/relative/path"));

        let local = "http://127.0.0.1:8080";
        assert!(allows(local, "http://127.0.0.1:8080/v1/join"));
        assert!(!allows(local, "http://127.0.0.1/v1/join"));
        assert!(!allows(local, "http://localhost:8080/v1/join"));
    }

    #[test]
    fn rejects_unusable_coordinator_urls() {
        assert!(Origin::parse("ftp://example.com").is_err());
        assert!(Origin::parse("example.com").is_err());
    }
}
