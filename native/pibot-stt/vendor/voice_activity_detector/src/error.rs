#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("unsupported VAD configuration: sample_rate={sample_rate}, chunk_size={chunk_size}. Only 8kHz/256, 16kHz/512 are allowed")]
    VadConfigError { sample_rate: i64, chunk_size: usize },
}
