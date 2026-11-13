//! Where the controls page draws each controller type: the body outlines, every button and the
//! sticks, in the diagram's 380 × 250 canvas. Pro and GameCube controllers are one body; Joy-Con
//! types place a left and a right Joy-Con (on their own, side by side, or on the console).

use egui::{CornerRadius, Pos2, Rect, pos2, vec2};
use lichii_common::settings::{ControllerType, NpadButton};

pub const SIZE: [f32; 2] = [380.0, 250.0];

pub enum Body {
    Shell(Rect, CornerRadius),
    Screen(Rect),
}

pub struct Stick {
    pub right: bool,
    pub center: Pos2,
    pub radius: f32,
}

pub struct Layout {
    pub bodies: Vec<Body>,
    pub buttons: Vec<(NpadButton, Rect)>,
    pub sticks: Vec<Stick>,
}

/// A Joy-Con is 80 wide, with its triggers above a 200 tall shell starting 40 down.
const JOYCON_WIDTH: f32 = 80.0;

pub fn layout(controller: ControllerType) -> Layout {
    use ControllerType::*;
    let mut layout = Layout {
        bodies: Vec::new(),
        buttons: Vec::new(),
        sticks: Vec::new(),
    };
    match controller {
        ProController | Switch2ProController | GameCube => full(&mut layout, controller),
        DualJoycons | Switch2DualJoycons => {
            left_joycon(&mut layout, 95.0, true);
            right_joycon(&mut layout, 205.0, true);
        }
        Handheld | Switch2Handheld => {
            left_joycon(&mut layout, 20.0, false);
            layout.bodies.push(Body::Screen(Rect::from_min_max(
                pos2(104.0, 52.0),
                pos2(276.0, 228.0),
            )));
            right_joycon(&mut layout, 280.0, false);
        }
        LeftJoycon | Switch2LeftJoycon => left_joycon(&mut layout, 150.0, true),
        RightJoycon | Switch2RightJoycon => right_joycon(&mut layout, 150.0, true),
    }
    layout
        .buttons
        .retain(|&(button, _)| controller.has_button(button) || analog_trigger(controller, button));
    layout
}

/// GameCube ZL/ZR are analog triggers: drawn, though never digital buttons.
pub fn analog_trigger(controller: ControllerType, button: NpadButton) -> bool {
    controller.has_analog_triggers() && matches!(button, NpadButton::ZL | NpadButton::ZR)
}

fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
    Rect::from_center_size(pos2(x, y), vec2(w, h))
}

fn full(layout: &mut Layout, controller: ControllerType) {
    use NpadButton::*;
    layout.bodies.push(Body::Shell(
        Rect::from_min_max(pos2(30.0, 46.0), pos2(350.0, 210.0)),
        CornerRadius::same(70),
    ));
    if controller == ControllerType::GameCube {
        for x in [70.0, 310.0] {
            layout.bodies.push(Body::Shell(
                Rect::from_center_size(pos2(x, 205.0), vec2(70.0, 70.0)),
                CornerRadius::same(35),
            ));
        }
    }
    let at = [
        (ZL, 85.0, 12.0, 70.0, 16.0),
        (L, 85.0, 32.0, 70.0, 16.0),
        (ZR, 295.0, 12.0, 70.0, 16.0),
        (R, 295.0, 32.0, 70.0, 16.0),
        (Minus, 160.0, 75.0, 18.0, 18.0),
        (Plus, 220.0, 75.0, 18.0, 18.0),
        (Capture, 160.0, 105.0, 16.0, 16.0),
        (Home, 220.0, 105.0, 20.0, 20.0),
        (C, 190.0, 135.0, 18.0, 18.0),
        (DPadUp, 135.0, 140.0, 18.0, 18.0),
        (DPadDown, 135.0, 180.0, 18.0, 18.0),
        (DPadLeft, 115.0, 160.0, 18.0, 18.0),
        (DPadRight, 155.0, 160.0, 18.0, 18.0),
        (X, 285.0, 72.0, 22.0, 22.0),
        (B, 285.0, 118.0, 22.0, 22.0),
        (Y, 262.0, 95.0, 22.0, 22.0),
        (A, 308.0, 95.0, 22.0, 22.0),
        (LStick, 105.0, 132.0, 30.0, 10.0),
        (RStick, 245.0, 197.0, 30.0, 10.0),
        (GL, 120.0, 232.0, 44.0, 14.0),
        (GR, 260.0, 232.0, 44.0, 14.0),
    ];
    for (button, x, y, w, h) in at {
        layout.buttons.push((button, rect(x, y, w, h)));
    }
    layout.sticks.push(Stick {
        right: false,
        center: pos2(105.0, 95.0),
        radius: 27.0,
    });
    layout.sticks.push(Stick {
        right: true,
        center: pos2(245.0, 160.0),
        radius: 27.0,
    });
}

/// A left Joy-Con whose shell starts at `x`; `rails` shows SL and SR on its right edge.
fn left_joycon(layout: &mut Layout, x: f32, rails: bool) {
    use NpadButton::*;
    layout.bodies.push(Body::Shell(
        Rect::from_min_max(pos2(x, 40.0), pos2(x + JOYCON_WIDTH, 240.0)),
        CornerRadius {
            nw: 36,
            sw: 36,
            ne: 6,
            se: 6,
        },
    ));
    let mut at = vec![
        (ZL, 40.0, 10.0, 60.0, 14.0),
        (L, 40.0, 28.0, 60.0, 14.0),
        (Minus, 62.0, 56.0, 16.0, 16.0),
        (LStick, 38.0, 124.0, 30.0, 10.0),
        (DPadUp, 38.0, 148.0, 18.0, 18.0),
        (DPadDown, 38.0, 188.0, 18.0, 18.0),
        (DPadLeft, 18.0, 168.0, 18.0, 18.0),
        (DPadRight, 58.0, 168.0, 18.0, 18.0),
        (Capture, 56.0, 220.0, 14.0, 14.0),
    ];
    if rails {
        at.push((LeftSL, 87.0, 90.0, 10.0, 30.0));
        at.push((LeftSR, 87.0, 190.0, 10.0, 30.0));
    }
    for (button, bx, y, w, h) in at {
        layout.buttons.push((button, rect(x + bx, y, w, h)));
    }
    layout.sticks.push(Stick {
        right: false,
        center: pos2(x + 38.0, 92.0),
        radius: 22.0,
    });
}

/// A right Joy-Con whose shell starts at `x`; `rails` shows SR and SL on its left edge.
fn right_joycon(layout: &mut Layout, x: f32, rails: bool) {
    use NpadButton::*;
    layout.bodies.push(Body::Shell(
        Rect::from_min_max(pos2(x, 40.0), pos2(x + JOYCON_WIDTH, 240.0)),
        CornerRadius {
            nw: 6,
            sw: 6,
            ne: 36,
            se: 36,
        },
    ));
    let mut at = vec![
        (ZR, 40.0, 10.0, 60.0, 14.0),
        (R, 40.0, 28.0, 60.0, 14.0),
        (Plus, 18.0, 56.0, 16.0, 16.0),
        (X, 40.0, 74.0, 20.0, 20.0),
        (B, 40.0, 122.0, 20.0, 20.0),
        (Y, 18.0, 98.0, 20.0, 20.0),
        (A, 62.0, 98.0, 20.0, 20.0),
        (RStick, 40.0, 198.0, 30.0, 10.0),
        (Home, 24.0, 222.0, 18.0, 18.0),
        (C, 58.0, 222.0, 16.0, 16.0),
    ];
    if rails {
        at.push((RightSR, -7.0, 90.0, 10.0, 30.0));
        at.push((RightSL, -7.0, 190.0, 10.0, 30.0));
    }
    for (button, bx, y, w, h) in at {
        layout.buttons.push((button, rect(x + bx, y, w, h)));
    }
    layout.sticks.push(Stick {
        right: true,
        center: pos2(x + 40.0, 166.0),
        radius: 22.0,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_type_draws_exactly_its_buttons_inside_the_canvas_without_overlap() {
        let canvas = Rect::from_min_size(Pos2::ZERO, SIZE.into());
        for controller in ControllerType::ALL {
            let layout = layout(controller);
            for button in NpadButton::ALL {
                let shown = layout.buttons.iter().filter(|(b, _)| *b == button).count();
                let expected = controller.has_button(button) || analog_trigger(controller, button);
                assert_eq!(shown, usize::from(expected), "{controller:?} {button:?}");
            }
            for (i, (a, ra)) in layout.buttons.iter().enumerate() {
                assert!(canvas.contains_rect(*ra), "{controller:?} {a:?} outside");
                for (b, rb) in &layout.buttons[i + 1..] {
                    assert!(!ra.intersects(*rb), "{controller:?} {a:?} overlaps {b:?}");
                }
            }
            let sticks = |right: bool| layout.sticks.iter().filter(|s| s.right == right).count();
            assert_eq!(
                sticks(false),
                usize::from(controller.has_left_stick()),
                "{controller:?}"
            );
            assert_eq!(
                sticks(true),
                usize::from(controller.has_right_stick()),
                "{controller:?}"
            );
        }
    }
}
