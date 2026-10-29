//! Minimal blocking HTTP client over `wasi:http/outgoing-handler`.
//!
//! TLS, DNS and the destination allowlist are all handled by the host, so the
//! worker carries no crypto or networking code of its own.

use std::io::{Read, Write};
use wasip2::http::outgoing_handler;
use wasip2::http::types::{Fields, Method, OutgoingBody, OutgoingRequest, Scheme};

pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

pub struct Client {
    https: bool,
    authority: String,
    base_path: String,
}

impl Client {
    /// `base` is the coordinator URL, e.g. `https://coordinator.example.com`.
    pub fn new(base: &str) -> Result<Self, String> {
        let (https, rest) = if let Some(rest) = base.strip_prefix("https://") {
            (true, rest)
        } else if let Some(rest) = base.strip_prefix("http://") {
            (false, rest)
        } else {
            return Err(format!(
                "coordinator URL must start with http:// or https://: {base}"
            ));
        };
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        if authority.is_empty() {
            return Err(format!("coordinator URL has no host: {base}"));
        }
        Ok(Self {
            https,
            authority: authority.to_string(),
            base_path: format!("/{}", path.trim_end_matches('/'))
                .trim_end_matches('/')
                .to_string(),
        })
    }

    pub fn get(&self, path: &str) -> Result<Response, String> {
        self.send(Method::Get, path, None)
    }

    pub fn post(&self, path: &str, content_type: &str, body: &[u8]) -> Result<Response, String> {
        self.send(Method::Post, path, Some((content_type, body)))
    }

    fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<(&str, &[u8])>,
    ) -> Result<Response, String> {
        let what = format!("{method:?} {path}");
        let err = |e: &dyn std::fmt::Debug| format!("{what}: {e:?}");

        let mut headers = Vec::new();
        if let Some((content_type, _)) = body {
            headers.push(("content-type".to_string(), content_type.as_bytes().to_vec()));
        }
        let request = OutgoingRequest::new(Fields::from_list(&headers).map_err(|e| err(&e))?);
        request.set_method(&method).map_err(|e| err(&e))?;
        let scheme = if self.https {
            Scheme::Https
        } else {
            Scheme::Http
        };
        request.set_scheme(Some(&scheme)).map_err(|e| err(&e))?;
        request
            .set_authority(Some(&self.authority))
            .map_err(|e| err(&e))?;
        let full_path = format!("{}{path}", self.base_path);
        request
            .set_path_with_query(Some(&full_path))
            .map_err(|e| err(&e))?;

        let outgoing_body = request.body().map_err(|e| err(&e))?;
        let pending = outgoing_handler::handle(request, None).map_err(|e| err(&e))?;
        if let Some((_, data)) = body {
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
        let incoming = response.consume().map_err(|e| err(&e))?;
        let mut body = Vec::new();
        incoming
            .stream()
            .map_err(|e| err(&e))?
            .read_to_end(&mut body)
            .map_err(|e| err(&e))?;
        Ok(Response { status, body })
    }
}
