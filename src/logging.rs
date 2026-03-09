use std::path::PathBuf;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub fn init_logging(log_level: &str, log_file: Option<PathBuf>) -> Option<WorkerGuard> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));

    if let Some(file_path) = log_file {
        let file_appender = tracing_appender::rolling::daily(
            file_path.parent().unwrap_or(&PathBuf::from(".")),
            file_path.file_name().unwrap().to_string_lossy().as_ref(),
        );
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(
                fmt::layer()
                    .with_writer(non_blocking)
                    .with_ansi(false)
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_file(true)
                    .with_line_number(true),
            )
            .with(
                fmt::layer()
                    .with_writer(std::io::stdout)
                    .with_ansi(true)
                    .with_target(true),
            )
            .init();

        Some(guard)
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(
                fmt::layer()
                    .with_writer(std::io::stdout)
                    .with_ansi(true)
                    .with_target(true),
            )
            .init();

        None
    }
}
