use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Layout},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, List, ListItem, Paragraph},
};
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::api;
use vecbox_core::{download, utils};

const MODELS: &[&str] = &[
    "alpaim/Qwen3-VL-Embedding-2B-GGUF-vecBox",
    "< Enter custom repo >",
];

#[derive(Clone)]
pub struct AppState {
    pub repo: String,
    pub quant: String,
    pub host: String,
    pub port: u16,
    pub max_pixels: Option<usize>,
    pub min_pixels: Option<usize>,
    pub server_running: Arc<AtomicBool>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            repo: "alpaim/Qwen3-VL-Embedding-2B-GGUF-vecBox".to_string(),
            quant: "Q4_K_M".to_string(),
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_pixels: None,
            min_pixels: None,
            server_running: Arc::new(AtomicBool::new(false)),
        }
    }
}

pub enum Step {
    Welcome,
    ModelSelection,
    QuantSelection,
    ServerConfig,
    Summary,
    Running,
}

pub struct State {
    pub step: Step,
    pub app: AppState,
    pub selected: usize,
    pub quants: Vec<String>,
    pub custom_repo: String,
    pub host: String,
    pub port: String,
    pub max_pixels: String,
    pub min_pixels: String,
    pub field: usize,
}

impl Default for State {
    fn default() -> Self {
        Self {
            step: Step::Welcome,
            app: AppState::default(),
            selected: 0,
            quants: Vec::new(),
            custom_repo: String::new(),
            host: "0.0.0.0".to_string(),
            port: "8080".to_string(),
            max_pixels: String::new(),
            min_pixels: String::new(),
            field: 0,
        }
    }
}

pub fn run_wizard() -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = State::default();
    let res = run_loop(&mut terminal, &mut state);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        eprintln!("Error: {}", err);
    }
    Ok(())
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut State,
) -> anyhow::Result<()> {
    loop {
        terminal.draw(|f| render(f, state))?;

        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                handle_input(key, state)?;
            }
        }

        if matches!(state.step, Step::Running) {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

fn handle_input(key: KeyEvent, state: &mut State) -> anyhow::Result<()> {
    match state.step {
        Step::Welcome => match key.code {
            KeyCode::Enter => state.step = Step::ModelSelection,
            _ => {}
        },
        Step::ModelSelection => match key.code {
            KeyCode::Up => {
                if state.selected > 0 {
                    state.selected -= 1
                }
            }
            KeyCode::Down => {
                if state.selected < MODELS.len() - 1 {
                    state.selected += 1
                }
            }
            KeyCode::Enter => {
                if state.selected == 1 {
                    state.step = Step::ModelSelection;
                } else {
                    state.app.repo = MODELS[state.selected].to_string();
                    fetch_quants(state)?;
                    state.selected = 0;
                    state.step = Step::QuantSelection;
                }
            }
            KeyCode::Esc => state.step = Step::Welcome,
            _ => {}
        },
        Step::QuantSelection => match key.code {
            KeyCode::Up => {
                if state.selected > 0 {
                    state.selected -= 1
                }
            }
            KeyCode::Down => {
                if state.selected < state.quants.len().saturating_sub(1) {
                    state.selected += 1
                }
            }
            KeyCode::Enter => {
                if !state.quants.is_empty() {
                    state.app.quant = state.quants[state.selected].clone();
                    state.field = 0;
                    state.step = Step::ServerConfig;
                }
            }
            KeyCode::Esc => {
                state.quants.clear();
                state.step = Step::ModelSelection;
            }
            _ => {}
        },
        Step::ServerConfig => match key.code {
            KeyCode::Up => {
                if state.field > 0 {
                    state.field -= 1
                }
            }
            KeyCode::Down => {
                if state.field < 3 {
                    state.field += 1
                }
            }
            KeyCode::Char(c) => match state.field {
                0 => state.host.push(c),
                1 => {
                    if c.is_ascii_digit() {
                        state.port.push(c);
                    }
                }
                2 => {
                    if c.is_ascii_digit() {
                        state.max_pixels.push(c);
                    }
                }
                3 => {
                    if c.is_ascii_digit() {
                        state.min_pixels.push(c);
                    }
                }
                _ => {}
            },
            KeyCode::Backspace => match state.field {
                0 => {
                    state.host.pop();
                }
                1 => {
                    state.port.pop();
                }
                2 => {
                    state.max_pixels.pop();
                }
                3 => {
                    state.min_pixels.pop();
                }
                _ => {}
            },
            KeyCode::Enter => {
                state.app.host = state.host.clone();
                state.app.port = state.port.parse().unwrap_or(8080);
                state.app.max_pixels = state.max_pixels.parse().ok();
                state.app.min_pixels = state.min_pixels.parse().ok();
                state.step = Step::Summary;
            }
            KeyCode::Esc => state.step = Step::QuantSelection,
            _ => {}
        },
        Step::Summary => match key.code {
            KeyCode::Enter => {
                start_server(state)?;
                state.step = Step::Running;
            }
            KeyCode::Esc => state.step = Step::ServerConfig,
            _ => {}
        },
        Step::Running => match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                state.app.server_running.store(false, Ordering::SeqCst);
                state.step = Step::Summary;
            }
            _ => {}
        },
    }
    Ok(())
}

fn fetch_quants(state: &mut State) -> anyhow::Result<()> {
    state.quants = download::fetch_available_quants(&state.app.repo)?;
    Ok(())
}

fn start_server(state: &mut State) -> anyhow::Result<()> {
    let app = state.app.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            if let Err(e) = run_server(app).await {
                eprintln!("Server error: {}", e);
            }
        });
    });
    std::thread::sleep(Duration::from_millis(500));
    Ok(())
}

async fn run_server(app: AppState) -> anyhow::Result<()> {
    use std::sync::Arc as StdArc;
    use vecbox_core::models::qwen3::Qwen3VLEmbedding;

    let downloaded = download::download_model(&app.repo, &app.quant)?;
    let device = utils::get_device()?;
    let dtype = utils::get_device_dtype(&device)?;

    let mut embedder = Qwen3VLEmbedding::from_gguf_and_mmproj(
        &downloaded.gguf_path,
        &downloaded.mmproj_path,
        &downloaded.config_dir,
        &device,
        dtype,
    )?;

    if let Some(v) = app.max_pixels {
        embedder.set_max_pixels(v);
    }
    if let Some(v) = app.min_pixels {
        embedder.set_min_pixels(v);
    }

    let api_state = api::AppState {
        embedder: StdArc::new(embedder),
        model_name: format!("{}-{}", app.repo, app.quant),
    };

    api::run_server(api_state, &app.host, app.port).await
}

fn render(frame: &mut Frame, state: &mut State) {
    let area = frame.area();

    let block = Block::default()
        .borders(Borders::ALL)
        .title(match state.step {
            Step::Welcome => " vecBox ",
            Step::ModelSelection => " Select Model ",
            Step::QuantSelection => " Select Quantization ",
            Step::ServerConfig => " Server Configuration ",
            Step::Summary => " Summary ",
            Step::Running => " Server Running ",
        });

    let inner =
        Layout::vertical([Constraint::Min(3), Constraint::Length(2)]).split(block.inner(area));

    frame.render_widget(block, area);

    match state.step {
        Step::Welcome => render_welcome(frame, inner[0]),
        Step::ModelSelection => render_model_list(frame, inner[0], state),
        Step::QuantSelection => render_quant_list(frame, inner[0], state),
        Step::ServerConfig => render_config_form(frame, inner[0], state),
        Step::Summary => render_summary(frame, inner[0], state),
        Step::Running => render_running(frame, inner[0], state),
    }

    let hint = match state.step {
        Step::Welcome => "ENTER: Start",
        Step::ModelSelection => "↑/↓: Select | ENTER: Confirm | ESC: Back",
        Step::QuantSelection => "↑/↓: Select | ENTER: Confirm | ESC: Back",
        Step::ServerConfig => "↑/↓: Navigate | ENTER: Continue | ESC: Back",
        Step::Summary => "ENTER: Start Server | ESC: Back",
        Step::Running => "Q/ESC: Stop",
    };
    frame.render_widget(Paragraph::new(hint).alignment(Alignment::Center), inner[1]);
}

fn render_welcome(frame: &mut Frame, area: ratatui::layout::Rect) {
    let chunks = Layout::vertical([
        Constraint::Percentage(40),
        Constraint::Percentage(20),
        Constraint::Percentage(40),
    ])
    .split(area);

    frame.render_widget(
        Paragraph::new("vecBox")
            .style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .alignment(Alignment::Center),
        chunks[0],
    );

    frame.render_widget(
        Paragraph::new("Multimodal Embedding Server")
            .style(Style::default().fg(Color::LightBlue))
            .alignment(Alignment::Center),
        chunks[1],
    );
}

fn render_model_list(frame: &mut Frame, area: ratatui::layout::Rect, state: &State) {
    let items: Vec<ListItem> = MODELS
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let prefix = if i == state.selected { "> " } else { "  " };
            let style = if i == state.selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(format!("{}{}", prefix, name)).style(style)
        })
        .collect();

    frame.render_widget(List::new(items), area);
}

fn render_quant_list(frame: &mut Frame, area: ratatui::layout::Rect, state: &State) {
    if state.quants.is_empty() {
        frame.render_widget(
            Paragraph::new("Loading...").alignment(Alignment::Center),
            area,
        );
        return;
    }

    let items: Vec<ListItem> = state
        .quants
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let prefix = if i == state.selected { "> " } else { "  " };
            let style = if i == state.selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(format!("{}{}", prefix, name)).style(style)
        })
        .collect();

    frame.render_widget(List::new(items), area);
}

fn render_config_form(frame: &mut Frame, area: ratatui::layout::Rect, state: &State) {
    let fields = [
        ("Host", &state.host),
        ("Port", &state.port),
        ("Max Pixels", &state.max_pixels),
        ("Min Pixels", &state.min_pixels),
    ];

    let rows: Vec<Line> = fields
        .iter()
        .enumerate()
        .map(|(i, (label, value))| {
            let cursor = if i == state.field { ">" } else { " " };
            let style = if i == state.field {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            Line::styled(
                format!(
                    "{} {}: {}",
                    cursor,
                    label,
                    if value.is_empty() {
                        "(optional)"
                    } else {
                        value
                    }
                ),
                style,
            )
        })
        .collect();

    frame.render_widget(Paragraph::new(rows), area);
}

fn render_summary(frame: &mut Frame, area: ratatui::layout::Rect, state: &State) {
    let lines = vec![
        Line::styled(format!("Repo:  {}", state.app.repo), Style::default()),
        Line::styled(format!("Quant: {}", state.app.quant), Style::default()),
        Line::default(),
        Line::styled(format!("Host:  {}", state.app.host), Style::default()),
        Line::styled(format!("Port:  {}", state.app.port), Style::default()),
        Line::styled(
            format!(
                "Max Pixels: {}",
                state
                    .app
                    .max_pixels
                    .map_or("(none)".into(), |v| v.to_string())
            ),
            Style::default(),
        ),
        Line::styled(
            format!(
                "Min Pixels: {}",
                state
                    .app
                    .min_pixels
                    .map_or("(none)".into(), |v| v.to_string())
            ),
            Style::default(),
        ),
    ];

    let chunks =
        Layout::vertical([Constraint::Percentage(60), Constraint::Percentage(40)]).split(area);

    frame.render_widget(Paragraph::new(lines), chunks[0]);

    frame.render_widget(
        Paragraph::new("[ Start Server ]")
            .style(
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
            .alignment(Alignment::Center),
        chunks[1],
    );
}

fn render_running(frame: &mut Frame, area: ratatui::layout::Rect, state: &State) {
    let lines = vec![
        Line::styled(
            "Status: RUNNING",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Line::default(),
        Line::styled(
            format!("URL: http://{}:{}", state.app.host, state.app.port),
            Style::default(),
        ),
        Line::default(),
        Line::styled(format!("Model: {}", state.app.repo), Style::default()),
        Line::styled(format!("Quant: {}", state.app.quant), Style::default()),
    ];

    frame.render_widget(Paragraph::new(lines), area);
}
