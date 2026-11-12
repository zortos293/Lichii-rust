//! Stick math: radial deadzone, range and modifier, in host units (-1..=1, +y up).

/// Applies a radial deadzone and range to a raw analog stick, as Lichii's `SanitizeStick`:
/// inside `deadzone` the stick reads zero, the rest of the travel is rescaled to start at zero,
/// `range` is the deflection that maps to full output (below 1 makes a worn stick reach the
/// edge), and the result is clamped to the unit circle.
pub fn shape_analog(x: f32, y: f32, deadzone: f32, range: f32) -> (f32, f32) {
    let x = if x.is_finite() { x } else { 0.0 };
    let y = if y.is_finite() { y } else { 0.0 };
    let r = x.hypot(y);
    if r <= deadzone || deadzone >= 1.0 || r == 0.0 {
        return (0.0, 0.0);
    }
    let factor = (r - deadzone) / (1.0 - deadzone) / r / range.max(f32::EPSILON);
    clamp_unit(x * factor, y * factor)
}

/// A stick driven by four digital directions. Opposite directions cancel, diagonals stay on the
/// unit circle, and the modifier scales the deflection (walk instead of run).
pub fn digital(up: bool, down: bool, left: bool, right: bool, scale: f32) -> (f32, f32) {
    let x = f32::from(right as u8) - f32::from(left as u8);
    let y = f32::from(up as u8) - f32::from(down as u8);
    if x == 0.0 && y == 0.0 {
        return (0.0, 0.0);
    }
    let r = x.hypot(y);
    (x / r * scale, y / r * scale)
}

pub fn clamp_unit(x: f32, y: f32) -> (f32, f32) {
    let r = x.hypot(y);
    if r > 1.0 { (x / r, y / r) } else { (x, y) }
}

/// Host units to Switch units.
pub fn to_switch(v: f32) -> i32 {
    (v.clamp(-1.0, 1.0) * super::npad::STICK_MAX as f32).round() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: (f32, f32), b: (f32, f32)) -> bool {
        (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5
    }

    #[test]
    fn inside_the_deadzone_is_zero() {
        assert_eq!(shape_analog(0.1, 0.1, 0.15, 1.0), (0.0, 0.0));
        assert_eq!(shape_analog(0.5, 0.5, 1.0, 1.0), (0.0, 0.0));
        assert_eq!(shape_analog(f32::NAN, 0.0, 0.0, 1.0), (0.0, 0.0));
    }

    #[test]
    fn deadzone_rescales_the_rest_of_the_travel() {
        let (x, y) = shape_analog(0.575, 0.0, 0.15, 1.0);
        assert!((x - 0.5).abs() < 1e-5 && y == 0.0, "{x}");
        assert!(close(shape_analog(1.0, 0.0, 0.15, 1.0), (1.0, 0.0)));
        assert!(close(shape_analog(0.0, -1.0, 0.0, 1.0), (0.0, -1.0)));
    }

    #[test]
    fn direction_is_kept() {
        let (x, y) = shape_analog(0.3, 0.4, 0.1, 1.0);
        assert!((y / x - 4.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn range_below_one_reaches_the_edge_early() {
        assert!(close(shape_analog(0.5, 0.0, 0.0, 0.5), (1.0, 0.0)));
        assert!(close(shape_analog(0.75, 0.0, 0.0, 1.5), (0.5, 0.0)));
    }

    #[test]
    fn output_is_clamped_to_the_unit_circle() {
        let (x, y) = shape_analog(1.0, 1.0, 0.0, 1.0);
        assert!((x.hypot(y) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn digital_directions() {
        assert_eq!(digital(false, false, false, false, 1.0), (0.0, 0.0));
        assert_eq!(digital(true, true, false, false, 1.0), (0.0, 0.0));
        assert_eq!(digital(true, false, false, false, 1.0), (0.0, 1.0));
        assert_eq!(digital(false, false, true, false, 0.5), (-0.5, 0.0));
        let (x, y) = digital(true, false, false, true, 1.0);
        assert!((x.hypot(y) - 1.0).abs() < 1e-5 && x > 0.0 && y > 0.0);
    }

    #[test]
    fn switch_units() {
        assert_eq!(to_switch(1.0), 0x7fff);
        assert_eq!(to_switch(-2.0), -0x7fff);
        assert_eq!(to_switch(0.0), 0);
    }
}
