//! Minimal blocking HTTP client over `wasi:http/outgoing-handler`.
//!
//! TLS, DNS and the destination allowlist are all handled by the host, so the
//! worker carries no crypto or networking code of its own. The worker may
//! name any URL; the runner refuses those outside the coordinator and the
//! volunteer's data origins.

use std::io::{Read, Write};
use wasip2::http::outgoing_handler;
use wasip2::http::types::{Fields, Method, OutgoingBody, OutgoingRequest, Scheme};

/// Redirects followed for one data download (Hugging Face sends one, to its CDN).
const MAX_REDIRECTS: usize = 5;

pub struct Response {
    pub status: u16,
    pub location: Option<String>,
    pub body: Vec<u8>,
}

/// Scheme, host and path of an absolute `http(s)://` URL.
struct Target {
    https: bool,
    authority: String,
    path: String,
}

impl Target {
    fn parse(url: &str) -> Result<Self, String> {
        let (https, rest) = if let Some(rest) = url.strip_prefix("https://") {
            (true, rest)
        } else if let Some(rest) = url.strip_prefix("http://") {
            (false, rest)
        } else {
            return Err(format!("URL must start with http:// or https://: {url}"));
        };
        let (authority, path) = match rest.find('/') {
            Some(i) => rest.split_at(i),
            None => (rest, "/"),
        };
        if authority.is_empty() {
            return Err(format!("URL has no host: {url}"));
        }
        Ok(Self {
            https,
            authority: authority.to_string(),
            path: path.to_string(),
        })
    }

    /// Resolves a `Location` header against this URL.
    fn redirect(&self, location: &str) -> Result<Self, String> {
        if location.starts_with('/') {
            Ok(Self {
                https: self.https,
                authority: self.authority.clone(),
                path: location.to_string(),
            })
        } else {
            Self::parse(location)
        }
    }
}

pub struct Client {
    coordinator: Target,
}

impl Client {
    /// `base` is the coordinator URL, e.g. `https://coordinator.example.com`.
    pub fn new(base: &str) -> Result<Self, String> {
        let mut coordinator = Target::parse(base).map_err(|e| format!("coordinator URL: {e}"))?;
        coordinator.path = coordinator.path.trim_end_matches('/').to_string();
        Ok(Self { coordinator })
    }

    pub fn get(&self, path: &str) -> Result<Response, String> {
        send(&self.at(path), Method::Get, &[], None)
    }

    pub fn post(&self, path: &str, content_type: &str, body: &[u8]) -> Result<Response, String> {
        let headers = [("content-type", content_type.as_bytes().to_vec())];
        send(&self.at(path), Method::Post, &headers, Some(body))
    }

    /// Downloads bytes `offset..offset + len` of `url`, following redirects.
    pub fn get_range(&self, url: &str, offset: u64, len: u64) -> Result<Vec<u8>, String> {
        let Some(last) = (offset + len).checked_sub(1) else {
            return Ok(Vec::new());
        };
        let range = [("range", format!("bytes={offset}-{last}").into_bytes())];
        let mut target = Target::parse(url)?;
        for _ in 0..=MAX_REDIRECTS {
            let r = send(&target, Method::Get, &range, None)?;
            match (r.status, r.location) {
                (301 | 302 | 303 | 307 | 308, Some(location)) => {
                    target = target.redirect(&location)?;
                }
                (206, _) if r.body.len() as u64 == len => return Ok(r.body),
                (206, _) => {
                    return Err(format!(
                        "{url}: asked for {len} bytes, got {}",
                        r.body.len()
                    ));
                }
                (status, _) => {
                    let msg = String::from_utf8_lossy(&r.body[..r.body.len().min(200)]);
                    return Err(format!("{url}: HTTP {status} {}", msg.trim()));
                }
            }
        }
        Err(format!("{url}: more than {MAX_REDIRECTS} redirects"))
    }

    fn at(&self, path: &str) -> Target {
        Target {
            https: self.coordinator.https,
            authority: self.coordinator.authority.clone(),
            path: format!("{}{path}", self.coordinator.path),
        }
    }
}

fn send(
    target: &Target,
    method: Method,
    headers: &[(&str, Vec<u8>)],
    body: Option<&[u8]>,
) -> Result<Response, String> {
    let what = format!("{method:?} {}", target.path);
    let err = |e: &dyn std::fmt::Debug| format!("{what}: {e:?}");

    let headers: Vec<(String, Vec<u8>)> = headers
        .iter()
        .map(|(name, value)| (name.to_string(), value.clone()))
        .collect();
    let request = OutgoingRequest::new(Fields::from_list(&headers).map_err(|e| err(&e))?);
    request.set_method(&method).map_err(|e| err(&e))?;
    let scheme = if target.https {
        Scheme::Https
    } else {
        Scheme::Http
    };
    request.set_scheme(Some(&scheme)).map_err(|e| err(&e))?;
    request
        .set_authority(Some(&target.authority))
        .map_err(|e| err(&e))?;
    request
        .set_path_with_query(Some(&target.path))
        .map_err(|e| err(&e))?;

    let outgoing_body = request.body().map_err(|e| err(&e))?;
    let pending = outgoing_handler::handle(request, None).map_err(|e| err(&e))?;
    if let Some(data) = body {
        let mut stream = outgoing_body.write().map_err(|e| err(&e))?;
        stream.write_all(data).map_err(|e| err(&e))?;
        stream.flush().map_err(|e| err(&e))?;
    }
    OutgoingBody::finish(outgoing_body, None).map_err(|e| err(&e))?;

    pending.subscribe().block();
    let response = pending
        .get()
        .ok_or_else(|| err(&"response not ready"))?
        .map_err(|()| err(&"response already taken"))?
        .map_err(|e| err(&e))?;
    let status = response.status();
    let location = response
        .headers()
        .get("location")
        .into_iter()
        .next()
        .and_then(|v| String::from_utf8(v).ok());
    let incoming = response.consume().map_err(|e| err(&e))?;
    let mut body = Vec::new();
    incoming
        .stream()
        .map_err(|e| err(&e))?
        .read_to_end(&mut body)
        .map_err(|e| err(&e))?;
    Ok(Response {
        status,
        location,
        body,
    })
}
