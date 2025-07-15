module UI

# Include all UI components
include("color_theme.jl")
include("sparkline_graph.jl") 
include("gpu_monitor.jl")
include("keyboard_handler.jl")
include("progress_tracker.jl")
include("realtime_update.jl")
include("console_dashboard.jl")
include("gpu_dashboard_integration.jl")
include("terminal_compat.jl")
include("log_panel.jl")

# Export all public APIs
using .ColorTheme
using .SparklineGraph
using .GPUMonitor
using .KeyboardHandler
using .ProgressTracker
using .RealtimeUpdate
using .ConsoleDashboard
using .GPUDashboardIntegration
using .TerminalCompat
using .LogPanel

# Re-export main components
export ColorTheme, SparklineGraph, GPUMonitor, KeyboardHandler
export ProgressTracker, RealtimeUpdate, ConsoleDashboard
export GPUDashboardIntegration, TerminalCompat, LogPanel

# Re-export key functions from each module
# From ColorTheme
export ThemeConfig, ColorScheme, ThresholdConfig, MetricThreshold
export create_theme, get_color, interpolate_color, get_status_color
export apply_theme_color, format_with_status, create_gradient
export flash_alert, get_pattern_symbol, get_colorblind_pattern
export status_indicator, set_theme!, set_colorblind_mode!, add_threshold!

# From SparklineGraph
export Sparkline, add_value!, render_sparkline, create_sparkline
export render_unicode_graph, render_ascii_graph, render_braille_graph
export get_trend_indicator, format_current_value

# From GPUMonitor
export GPUMonitorState, start_monitoring!, stop_monitoring!
export get_gpu_metrics, get_historical_metrics, GPUPanelContent
export format_gpu_metrics, calculate_gpu_efficiency

# From KeyboardHandler
export KeyHandler, KeyBinding, KeyEvent, CommandContext
export register_handler!, unregister_handler!, handle_key!
export create_key_handler, set_context!, get_help_text
export QUIT_KEY, PAUSE_KEY, HELP_KEY, ESC_KEY

# From ProgressTracker
export ProgressState, ProgressEstimator, StageProgress
export update_progress!, calculate_eta, format_progress
export get_progress_bar, get_stage_indicator, reset_progress!
export pause_progress!, resume_progress!, is_complete

# From RealtimeUpdate
export UpdateManager, UpdateBuffer, PanelUpdate
export start_updates!, stop_updates!, update_panel!
export set_refresh_rate!, get_update_stats, reset_stats!
export create_update_manager, add_update_source!

# From ConsoleDashboard
export DashboardConfig, DashboardState, PanelConfig
export create_dashboard, start_dashboard!, stop_dashboard!
export update_dashboard_panel!, resize_dashboard!
export get_dashboard_metrics, save_dashboard_state

# From GPUDashboardIntegration
export GPUDashboardPanel, create_gpu_panels, update_gpu_panels!
export format_gpu_summary, get_gpu_alerts, sync_gpu_states

# From TerminalCompat
export TerminalCapabilities, detect_terminal, get_capabilities
export supports_unicode, supports_colors, supports_true_color
export get_terminal_size, clear_screen, move_cursor
export save_cursor, restore_cursor, hide_cursor, show_cursor

# From LogPanel
export LogLevel, LogEntry, CircularLogBuffer, LogPanelState, LogPanelContent
export add_log!, filter_logs, search_logs, export_logs, format_log_entry
export DEBUG, INFO, WARN, ERROR

end # module UI