def load_tasks_safely() -> List[Dict]:
    """Safely load tasks without any modifications"""
    global last_update
    
    taskmaster_dir = find_taskmaster_dir()
    if not taskmaster_dir:
        terminal_console.print("[red]❌ Cannot find .taskmaster/tasks/tasks.json[/red]")
        return []#!/usr/bin/env python3
"""
Task Master Read-Only Viewer - Safe viewer for .taskmaster/tasks/tasks.json
This script ONLY READS, never modifies your task files!
Place in: scripts/taskmaster-mirror-simple.py
Requires: pip install flask-socketio rich watchdog
"""

import json
import os
import sys
from pathlib import Path
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional

from flask import Flask, render_template_string
from flask_socketio import SocketIO
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add parent directory to path to find .taskmaster
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'taskmaster-viewer-readonly'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state (read-only copy of tasks)
current_tasks = []
last_update = None

# Console setup
console = Console(record=True, width=120, force_terminal=True)
terminal_console = Console()  # For terminal output

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Task Master Viewer (Read-Only)</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body {
            margin: 0;
            background: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .header {
            background: #161b22;
            padding: 10px 20px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .badge {
            background: #1f6feb;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        .safe-mode {
            background: #2ea043;
        }
        
        #console-mirror {
            padding: 20px;
            height: calc(100vh - 60px);
            overflow-y: auto;
        }
        
        .status {
            position: fixed;
            bottom: 10px;
            right: 10px;
            background: #161b22;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #30363d;
            font-size: 12px;
        }
        
        /* Make sure Rich HTML displays properly */
        pre {
            margin: 0;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h3 style="margin: 0;">📋 Task Master Viewer</h3>
        </div>
        <div>
            <span class="badge safe-mode">🔒 READ-ONLY MODE</span>
            <span class="badge">No modifications to your files</span>
        </div>
    </div>
    
    <div id="console-mirror"></div>
    
    <div class="status" id="status">
        Last update: <span id="last-update">Never</span>
    </div>
    
    <script>
        const socket = io();
        
        socket.on('console_update', (data) => {
            document.getElementById('console-mirror').innerHTML = data.html;
            document.getElementById('last-update').textContent = data.timestamp;
        });
        
        socket.on('connect', () => {
            console.log('Connected to Task Master Viewer');
        });
        
        // Auto-refresh
        setInterval(() => {
            socket.emit('request_refresh');
        }, 5000);
    </script>
</body>
</html>
'''

class TaskFileWatcher(FileSystemEventHandler):
    """Watch for changes in tasks.json"""
    def __init__(self, callback):
        self.callback = callback
        
    def on_modified(self, event):
        if event.src_path.endswith('tasks.json'):
            terminal_console.print("[yellow]📝 Detected change in tasks.json[/yellow]")
            self.callback()

def find_taskmaster_dir():
    """Find .taskmaster directory in project root"""
    current = Path.cwd()
    
    # Try current directory first
    if (current / '.taskmaster' / 'tasks' / 'tasks.json').exists():
        return current / '.taskmaster' / 'tasks'
    
    # Try parent directory (if script is in scripts/)
    parent = current.parent
    if (parent / '.taskmaster' / 'tasks' / 'tasks.json').exists():
        return parent / '.taskmaster' / 'tasks'
    
    # Try to find it relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    if (project_root / '.taskmaster' / 'tasks' / 'tasks.json').exists():
        return project_root / '.taskmaster' / 'tasks'
    
    return None

def load_tasks_safely() -> List[Dict]:
    """Safely load tasks without any modifications"""
    global last_update
    
    taskmaster_dir = find_taskmaster_dir()
    if not taskmaster_dir:
        terminal_console.print("[red]❌ Cannot find .taskmaster/tasks/tasks.json[/red]")
        return []
    
    tasks_file = taskmaster_dir / 'tasks.json'
    
    try:
        with open(tasks_file, 'r') as f:
            data = json.load(f)
        
        # Handle the nested structure - tasks are in master.tasks
        if isinstance(data, dict) and 'master' in data and 'tasks' in data['master']:
            tasks = data['master']['tasks']
        elif isinstance(data, list):
            # Fallback for direct list format
            tasks = data
        else:
            terminal_console.print("[red]❌ Unexpected tasks.json structure[/red]")
            return []
        
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        terminal_console.print(f"[green]✓ Loaded {len(tasks)} tasks from {tasks_file}[/green]")
        return tasks
    except Exception as e:
        terminal_console.print(f"[red]Error reading tasks: {e}[/red]")
        return []

def create_display() -> Layout:
    """Create Rich display layout"""
    layout = Layout()
    
    # Header
    header = Panel(
        Text("Task Master Viewer - Read-Only Mode", style="bold cyan", justify="center"),
        subtitle="📁 Viewing .taskmaster/tasks/tasks.json",
        style="cyan"
    )
    
    # Stats
    total = len(current_tasks)
    completed = len([t for t in current_tasks if t.get('status') == 'completed'])
    pending = total - completed
    progress = (completed / total * 100) if total > 0 else 0
    
    stats_text = f"""
[green]Completed:[/green] {completed}
[yellow]Pending:[/yellow] {pending}
[cyan]Total:[/cyan] {total}
[blue]Progress:[/blue] {progress:.1f}%
    """
    
    stats_panel = Panel(stats_text.strip(), title="📊 Statistics", border_style="blue")
    
    # Tasks table
    table = Table(title="Current Tasks", box=box.ROUNDED, show_header=True)
    table.add_column("ID", style="cyan", width=15)
    table.add_column("Title", style="white", width=50)
    table.add_column("Priority", width=10)
    table.add_column("Status", width=12)
    table.add_column("Subtasks", width=10)
    
    # Sort tasks: pending first, then by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_tasks = sorted(current_tasks, key=lambda t: (
        0 if t.get('status') != 'completed' else 1,
        priority_order.get(t.get('priority', 'medium'), 1),
        t.get('id', '')
    ))
    
    for task in sorted_tasks[:20]:  # Show first 20
        priority = task.get('priority', 'medium')
        priority_color = {
            'high': 'red',
            'medium': 'yellow', 
            'low': 'green'
        }.get(priority, 'white')
        
        status = task.get('status', 'pending')
        status_emoji = '✅' if status == 'completed' else '⏳'
        
        subtasks_count = len(task.get('subtasks', []))
        
        table.add_row(
            task.get('id', 'unknown'),
            Text(task.get('title', 'Untitled'), overflow="ellipsis"),
            f"[{priority_color}]{priority}[/{priority_color}]",
            f"{status_emoji} {status}",
            str(subtasks_count)
        )
    
    tasks_panel = Panel(table, title="📋 Tasks", border_style="green")
    
    # Combine layout
    layout.split(
        Layout(header, size=3),
        Layout(stats_panel, size=6),
        Layout(tasks_panel)
    )
    
    return layout

def update_displays():
    """Update both console and web displays"""
    global current_tasks
    
    # Reload tasks
    current_tasks = load_tasks_safely()
    
    # Create display
    console.clear()
    display = create_display()
    console.print(display)
    
    # Send to web
    html = console.export_html(inline_styles=True)
    socketio.emit('console_update', {
        'html': html,
        'timestamp': last_update or 'Never'
    })

def console_display_loop():
    """Main console display loop"""
    with Live(create_display(), console=terminal_console, refresh_per_second=2) as live:
        while True:
            live.update(create_display())
            time.sleep(2)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@socketio.on('request_refresh')
def handle_refresh():
    update_displays()

def start_file_watcher():
    """Start watching for file changes"""
    taskmaster_dir = find_taskmaster_dir()
    if not taskmaster_dir:
        return
    
    event_handler = TaskFileWatcher(update_displays)
    observer = Observer()
    observer.schedule(event_handler, str(taskmaster_dir), recursive=False)
    observer.start()
    
    terminal_console.print(f"[green]👁  Watching for changes in {taskmaster_dir}[/green]")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("📋 TASK MASTER VIEWER - READ-ONLY MODE")
    print("="*60)
    print("🔒 This script will NOT modify any files")
    print("📁 Reading from: .taskmaster/tasks/tasks.json")
    print("🌐 Web interface: http://localhost:5000")
    print("📺 Console: This terminal")
    print("="*60 + "\n")
    
    # Initial load
    current_tasks = load_tasks_safely()
    
    if not current_tasks:
        print("⚠️  No tasks found. Make sure .taskmaster/tasks/tasks.json exists")
        print("   Run from project root or scripts/ directory")
        sys.exit(1)
    
    # Start file watcher
    watcher_thread = threading.Thread(target=start_file_watcher, daemon=True)
    watcher_thread.start()
    
    # Start console display
    console_thread = threading.Thread(target=console_display_loop, daemon=True)
    console_thread.start()
    
    # Initial display update
    update_displays()
    
    # Start web server
    try:
        socketio.run(app, debug=False, port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Task Master Viewer...")
        sys.exit(0)