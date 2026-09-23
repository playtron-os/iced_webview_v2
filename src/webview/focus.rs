//! The view as a stop in the host's keyboard focus.
//!
//! The engine only hears keys the host forwards, and the host has to know when
//! to: this frame is the view's one focus stop — reached by Tab like any other
//! control, or taken by a click or a tap on the page — and while it holds focus
//! every key but Tab goes to the page, so the arrows, Page Up and Down, Space,
//! Home and End read the email and Enter follows a focused link. Tab is left
//! alone, to move on: a view that swallowed it would be a trap. Ctrl+C copies
//! the page's selection through [`Action::CopySelection`], as before.
//!
//! The frame also tells the engine when it gains and loses focus, which is what
//! an engine needs before it acts on keys at all (CEF's browser host focus).

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer;
use iced::advanced::widget::operation::Focusable;
use iced::advanced::widget::{self, Operation, Tree, Widget};
use iced::advanced::{overlay, Renderer as _, Shell};
use iced::keyboard::{self, key::Named, Key};
use iced::{mouse, touch, Border, Color, Element, Event, Length, Rectangle, Size, Vector};

use crate::webview::basic::Action;

type Renderer = iced::Renderer;

/// How the frame shows it has the keyboard — the host's own focus ring, so the
/// view marks focus the way every other control around it does.
#[derive(Debug, Clone, Copy)]
pub struct FocusRing {
    /// The ring's colour.
    pub color: Color,
    /// Its stroke, drawn just inside the view's edge.
    pub width: f32,
    /// The view's corner radius, which the ring follows.
    pub radius: f32,
    /// Whether rings are showing right now — a host that paints them only
    /// while the keyboard is driving (`:focus-visible`) answers that here.
    pub visible: fn() -> bool,
}

pub(crate) struct KeyboardFrame<'a, Theme> {
    content: Element<'a, Action, Theme, Renderer>,
    ring: Option<FocusRing>,
}

impl<'a, Theme> KeyboardFrame<'a, Theme> {
    pub(crate) fn new(
        content: Element<'a, Action, Theme, Renderer>,
        ring: Option<FocusRing>,
    ) -> Self {
        Self { content, ring }
    }
}

#[derive(Debug, Default)]
struct FrameState {
    focused: bool,
    /// What the engine was last told, so each change is told once.
    told: bool,
}

impl Focusable for FrameState {
    fn is_focused(&self) -> bool {
        self.focused
    }

    fn focus(&mut self) {
        self.focused = true;
    }

    fn unfocus(&mut self) {
        self.focused = false;
    }
}

impl<Theme> Widget<Action, Theme, Renderer> for KeyboardFrame<'_, Theme> {
    fn tag(&self) -> widget::tree::Tag {
        widget::tree::Tag::of::<FrameState>()
    }

    fn state(&self) -> widget::tree::State {
        widget::tree::State::new(FrameState::default())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_ref(&self.content));
    }

    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }

    fn size_hint(&self) -> Size<Length> {
        self.content.as_widget().size_hint()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.content
            .as_widget_mut()
            .layout(&mut tree.children[0], renderer, limits)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        operation.focusable(
            None,
            layout.bounds(),
            tree.state.downcast_mut::<FrameState>(),
        );
        self.content
            .as_widget_mut()
            .operate(&mut tree.children[0], layout, renderer, operation);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        shell: &mut Shell<'_, Action>,
        viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let state = tree.state.downcast_mut::<FrameState>();

        // A press on the page takes the keyboard; one anywhere else lets it go.
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(_)) => state.focused = cursor.is_over(bounds),
            Event::Touch(touch::Event::FingerPressed { position, .. }) => {
                state.focused = bounds.contains(*position);
            }
            _ => {}
        }
        if state.focused != state.told {
            state.told = state.focused;
            shell.publish(if state.focused {
                Action::Focus
            } else {
                Action::Unfocus
            });
        }

        if let (true, Event::Keyboard(key_event)) = (state.focused, event) {
            match key_event {
                keyboard::Event::KeyPressed {
                    key: Key::Named(Named::Tab),
                    ..
                }
                | keyboard::Event::KeyReleased {
                    key: Key::Named(Named::Tab),
                    ..
                } => {}
                keyboard::Event::KeyPressed {
                    key: Key::Character(c),
                    modifiers,
                    ..
                } if modifiers.command() && c.as_str() == "c" => {
                    shell.publish(Action::CopySelection);
                    shell.capture_event();
                }
                _ => {
                    shell.publish(Action::SendKeyboardEvent(key_event.clone()));
                    shell.capture_event();
                }
            }
        }

        // Always, keys included: the content keeps the modifiers its mouse and
        // touch events carry.
        self.content.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout,
            cursor,
            renderer,
            shell,
            viewport,
        );
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            layout,
            cursor,
            viewport,
        );
        let Some(ring) = self.ring else {
            return;
        };
        if !tree.state.downcast_ref::<FrameState>().focused || !(ring.visible)() {
            return;
        }
        // Its own layer: the page is a custom primitive, drawn after the
        // quads of the layer it shares, and a ring in that layer sat under it.
        let bounds = layout.bounds();
        renderer.with_layer(bounds, |renderer| {
            renderer.fill_quad(
                renderer::Quad {
                    bounds,
                    border: Border {
                        color: ring.color,
                        width: ring.width,
                        radius: ring.radius.into(),
                        ..Border::default()
                    },
                    ..renderer::Quad::default()
                },
                Color::TRANSPARENT,
            );
        });
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.content.as_widget().mouse_interaction(
            &tree.children[0],
            layout,
            cursor,
            viewport,
            renderer,
        )
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'b>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<overlay::Element<'b, Action, Theme, Renderer>> {
        self.content.as_widget_mut().overlay(
            &mut tree.children[0],
            layout,
            renderer,
            viewport,
            translation,
        )
    }
}

impl<'a, Theme: 'a> From<KeyboardFrame<'a, Theme>> for Element<'a, Action, Theme, Renderer> {
    fn from(frame: KeyboardFrame<'a, Theme>) -> Self {
        Element::new(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::widget::{row, Space};
    use iced::Point;
    use iced_test::simulator::click;
    use iced_test::Simulator;

    /// The frame on the left half of the window, something else on the right.
    fn scene() -> Element<'static, Action, iced::Theme, Renderer> {
        let page: Element<'static, Action, iced::Theme, Renderer> =
            Space::new().width(Length::Fill).height(Length::Fill).into();
        row![
            Element::from(KeyboardFrame::new(page, None)),
            Space::new().width(Length::Fill).height(Length::Fill),
        ]
        .into()
    }

    const WINDOW: Size = Size::new(800.0, 600.0);
    const ON_PAGE: Point = Point::new(200.0, 300.0);
    const OFF_PAGE: Point = Point::new(600.0, 300.0);

    fn press_at(ui: &mut Simulator<'_, Action>, at: Point) {
        ui.point_at(at);
        let _ = ui.simulate(click());
    }

    fn forwarded(messages: &[Action]) -> usize {
        messages
            .iter()
            .filter(|m| matches!(m, Action::SendKeyboardEvent(_)))
            .count()
    }

    #[test]
    fn a_press_on_the_page_gives_it_the_keyboard_and_tab_moves_on() {
        let mut ui = Simulator::with_size(iced_test::core::Settings::default(), WINDOW, scene());
        press_at(&mut ui, ON_PAGE);
        let _ = ui.tap_key(Key::Named(Named::ArrowDown));
        let _ = ui.tap_key(Key::Named(Named::Tab));

        let messages: Vec<_> = ui.into_messages().collect();
        assert!(
            matches!(messages.first(), Some(Action::Focus)),
            "the engine was not told the page has the keyboard: {messages:?}"
        );
        // ArrowDown pressed and released; Tab neither.
        assert_eq!(forwarded(&messages), 2, "{messages:?}");
    }

    #[test]
    fn focus_from_the_keyboard_is_the_same_focus() {
        use iced::advanced::widget::operation::{self, Operation as _};
        let mut ui = Simulator::with_size(iced_test::core::Settings::default(), WINDOW, scene());
        let mut op: Box<dyn Operation> = Box::new(operation::focusable::focus_next::<()>());
        loop {
            ui.operate(&mut op);
            match op.finish() {
                operation::Outcome::Chain(next) => op = next,
                _ => break,
            }
        }
        let _ = ui.tap_key(Key::Named(Named::PageDown));

        let messages: Vec<_> = ui.into_messages().collect();
        assert!(messages.iter().any(|m| matches!(m, Action::Focus)), "{messages:?}");
        assert_eq!(forwarded(&messages), 2, "{messages:?}");
    }

    #[test]
    fn a_press_elsewhere_takes_the_keyboard_back() {
        let mut ui = Simulator::with_size(iced_test::core::Settings::default(), WINDOW, scene());
        press_at(&mut ui, ON_PAGE);
        press_at(&mut ui, OFF_PAGE);
        let _ = ui.tap_key(Key::Named(Named::ArrowDown));

        let messages: Vec<_> = ui.into_messages().collect();
        assert!(messages.iter().any(|m| matches!(m, Action::Unfocus)), "{messages:?}");
        assert_eq!(forwarded(&messages), 0, "keys went to a page that lost focus: {messages:?}");
    }
}
