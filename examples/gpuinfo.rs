//! Prints what Chromium thinks of the GPU inside this embedding.
//!
//! `chrome://gpu` is the only authoritative answer to "is GPU compositing on",
//! and it has to be asked from inside the same process configuration the app
//! uses — the answer differs from a standalone Chromium's. Runs the engine
//! without a window, so nothing here depends on a renderer being up.
//!
//! Run with the same environment as the app, e.g.
//! `ICED_WEBVIEW_ACCELERATED=1 cargo run --features cef --example gpuinfo`.

use iced_webview::{Cef, Engine, PageType};

fn main() {
    if iced_webview::cef_subprocess_check() {
        return;
    }

    let mut engine = Cef::default();
    let id = engine.new_view(
        iced::Size::new(1024, 768),
        Some(PageType::Url("chrome://gpu".to_string())),
    );

    // Pump until the page has had time to render its report.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(12);
    while std::time::Instant::now() < deadline {
        engine.update();
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    // The feature table is what matters; the rest of the page is enormous.
    engine.evaluate_javascript(
        id,
        // chrome://gpu builds its report inside shadow roots, so a plain
        // innerText on the document misses all of it.
        r#"(() => {
             const out = [];
             const walk = (root) => {
               for (const el of root.querySelectorAll('*')) {
                 if (el.shadowRoot) walk(el.shadowRoot);
               }
               const t = root.textContent || '';
               if (t.trim()) out.push(t);
             };
             walk(document);
             return out.join('\n').replace(/\n{2,}/g, '\n').slice(0, 4000);
           })()"#,
        1,
    );

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(6);
    while std::time::Instant::now() < deadline {
        engine.update();
        if let Some(result) = engine.take_eval_results(id).into_iter().next() {
            println!("--- chrome://gpu ---\n{result:?}");
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    eprintln!("no result from chrome://gpu within the timeout");
}
