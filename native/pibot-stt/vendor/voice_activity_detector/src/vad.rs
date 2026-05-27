use std::sync::{Arc, LazyLock, Mutex};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::{Error, Sample};

#[derive(Debug)]
pub struct VoiceActivityDetector {
    session: Arc<Mutex<Session>>,
    chunk_size: usize,
    sample_rate: i64,
    state: ndarray::ArrayD<f32>,
}

const MODEL: &[u8] = include_bytes!("silero_vad.onnx");

static DEFAULT_SESSION: LazyLock<Arc<Mutex<Session>>> = LazyLock::new(|| {
    Arc::new(Mutex::new(
        Session::builder()
            .expect("failed to create ONNX Runtime session builder")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("failed to set ONNX Runtime optimization level")
            .with_intra_threads(1)
            .expect("failed to set ONNX Runtime intra threads")
            .with_inter_threads(1)
            .expect("failed to set ONNX Runtime inter threads")
            .commit_from_memory(MODEL)
            .expect("failed to load embedded Silero VAD model"),
    ))
});

impl VoiceActivityDetector {
    pub fn builder() -> VoiceActivityDetectorBuilder {
        VoiceActivityDetectorConfig::builder()
    }

    pub fn reset(&mut self) {
        self.state = ndarray::Array3::<f32>::zeros((2, 1, 128)).into_dyn();
    }

    pub fn predict<S, I>(&mut self, samples: I) -> f32
    where
        S: Sample,
        I: IntoIterator<Item = S>,
    {
        let mut input = ndarray::Array2::<f32>::zeros((1, self.chunk_size));
        for (index, sample) in samples.into_iter().take(self.chunk_size).enumerate() {
            input[[0, index]] = sample.to_f32();
        }

        let sample_rate = ndarray::arr0::<i64>(self.sample_rate);
        let state = std::mem::take(&mut self.state);
        let inputs = ort::inputs![
            Tensor::from_array(input).expect("failed to create VAD input tensor"),
            Tensor::from_array(state).expect("failed to create VAD state tensor"),
            Tensor::from_array(sample_rate).expect("failed to create VAD sample-rate tensor"),
        ];

        let mut session = self.session.lock().expect("VAD session lock poisoned");
        let outputs = session.run(inputs).expect("VAD inference failed");

        self.state = outputs["stateN"]
            .try_extract_array::<f32>()
            .expect("missing VAD state output")
            .to_owned();

        let output = outputs["output"]
            .try_extract_array::<f32>()
            .expect("missing VAD probability output");
        output[[0, 0]]
    }
}

#[derive(Debug, typed_builder::TypedBuilder)]
#[builder(
    builder_method(vis = ""),
    builder_type(name = VoiceActivityDetectorBuilder, vis = "pub"),
    build_method(into = Result<VoiceActivityDetector, Error>, vis = "pub")
)]
struct VoiceActivityDetectorConfig {
    #[builder(setter(into))]
    chunk_size: usize,
    #[builder(setter(into))]
    sample_rate: i64,
    #[builder(default, setter(strip_option))]
    session: Option<Arc<Mutex<Session>>>,
}

impl From<VoiceActivityDetectorConfig> for Result<VoiceActivityDetector, Error> {
    fn from(config: VoiceActivityDetectorConfig) -> Self {
        let chunk_size = match config.sample_rate {
            8000 => 256,
            16000 => 512,
            sample_rate => {
                return Err(Error::VadConfigError {
                    sample_rate,
                    chunk_size: config.chunk_size,
                });
            }
        };

        Ok(VoiceActivityDetector {
            session: config.session.unwrap_or_else(|| DEFAULT_SESSION.clone()),
            chunk_size,
            sample_rate: config.sample_rate,
            state: ndarray::Array3::<f32>::zeros((2, 1, 128)).into_dyn(),
        })
    }
}
