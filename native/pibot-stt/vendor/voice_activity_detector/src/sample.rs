pub trait Sample: Copy + Default + Sized {
    fn to_f32(self) -> f32;
}

impl Sample for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl Sample for i16 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 32768.0
    }
}

impl Sample for i8 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 128.0
    }
}

impl Sample for u16 {
    fn to_f32(self) -> f32 {
        (f32::from(self) - 32768.0) / 32768.0
    }
}

impl Sample for u8 {
    fn to_f32(self) -> f32 {
        (f32::from(self) - 128.0) / 128.0
    }
}
