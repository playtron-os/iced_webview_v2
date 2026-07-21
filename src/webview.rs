use iced::{mouse, touch};

/// Translate a touch event into the equivalent mouse event so a touchscreen can
/// drive the embedded browser engine, which only understands mouse input.
/// A finger press/lift maps to a left button press/release (a tap is a click);
/// a finger move maps to a cursor move. `FingerLost` (compositor cancel, e.g.
/// palm rejection) is treated as a release so the engine ends any drag state.
pub(crate) fn touch_to_mouse(event: &touch::Event) -> mouse::Event {
    match event {
        touch::Event::FingerPressed { .. } => mouse::Event::ButtonPressed(mouse::Button::Left),
        touch::Event::FingerMoved { position, .. } => mouse::Event::CursorMoved {
            position: *position,
        },
        touch::Event::FingerLifted { .. } | touch::Event::FingerLost { .. } => {
            mouse::Event::ButtonReleased(mouse::Button::Left)
        }
    }
}

/// Whether a touch event ends the finger's contact (lift or cancel).
pub(crate) fn is_touch_release(event: &touch::Event) -> bool {
    matches!(
        event,
        touch::Event::FingerLifted { .. } | touch::Event::FingerLost { .. }
    )
}

/// Advanced is a more complex interface than basic and assumes the user stores all the view ids themselves.
/// This gives the user more freedom by allowing them to view multiple views at the same time, but removes
/// actions like close current
pub mod advanced;
/// Basic allows users to have simple interfaces like close current and
/// allows users to index views by ints like 0, 1 , or 2
pub mod basic;

/// Shader-based rendering widget for engines that manage their own scrolling
/// (e.g. servo, cef). Uses direct GPU texture updates to avoid Handle cache churn.
#[cfg(any(feature = "servo", feature = "cef"))]
pub(crate) mod shader_widget;
