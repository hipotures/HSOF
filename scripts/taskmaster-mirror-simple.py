#!/usr/bin/env python3
"""
Task Master Mirror Mode - Simple console mirroring to web
Requires: pip install flask-socketio rich
"""

import threading
import time
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import io

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Shared console with StringIO buffer
console_buffer = io.StringIO()
console = Console(
    file=console_buffer,
    record=True,
    width=120,
    force_terminal=True,
    color_system="truecolor"
)

# Also create a console for terminal output
terminal_console = Console()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Task Master - Mirror Mode</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body {
            margin: 0;
            background: #000;
            font-family: monospace;
        }
        #mirror {
            padding: 20px;
        }
        /* Terminal colors */
        .ansi-black { color: #000000; }
        .ansi-red { color: #cd0000; }
        .ansi-green { color: #00cd00; }
        .ansi-yellow { color: #cdcd00; }
        .ansi-blue { color: #0000ee; }
        .ansi-magenta { color: #cd00cd; }
        .ansi-cyan { color: #00cdcd; }
        .ansi-white { color: #e5e5e5; }
        .ansi-bright-black { color: #7f7f7f; }
        .ansi-bright-red { color: #ff0000; }
        .ansi-bright-green { color: #00ff00; }
        .ansi-bright-yellow { color: #ffff00; }
        .ansi-bright-blue { color: #5c5cff; }
        .ansi-bright-magenta { color: #ff00ff; }
        .ansi-bright-cyan { color: #00ffff; }
        .ansi-bright-white { color: #ffffff; }
    </style>
</head>
<body>
    <div id="mirror"></div>
    <script>
        const socket = io();
        socket.on('console_update', (data) => {
            document.getElementById('mirror').innerHTML = data.html;
        });
    </script>
</body>
</html>
'''

def broadcast_console():
    """Send console content to all web clients"""
    html = console.export_html(inline_styles=True)
    socketio.emit('console_update', {'html': html})

def mirror_print(*args, **kwargs):
    """Print to both terminal and web"""
    # Print to terminal
    terminal_console.print(*args, **kwargs)
    
    # Clear buffer and print to web console
    console_buffer.truncate(0)
    console_buffer.seek(0)
    console.print(*args, **kwargs)
    
    # Broadcast to web
    broadcast_console()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def demo_task_execution():
    """Demo showing real-time sync"""
    time.sleep(1)
    
    # Welcome message
    mirror_print(Panel.fit(
        "[bold cyan]Task Master - Mirror Mode Demo[/bold cyan]\n"
        "Same output in terminal and web browser!",
        border_style="cyan"
    ))
    
    time.sleep(2)
    
    # Show task table
    table = Table(title="📋 Pending Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Status", style="yellow")
    
    tasks = [
        ("task-001", "Initialize project", "pending"),
        ("task-002", "Setup database", "pending"),
        ("task-003", "Create API", "pending"),
    ]
    
    for task_id, title, status in tasks:
        table.add_row(task_id, title, status)
    
    mirror_print(table)
    
    time.sleep(2)
    
    # Execute tasks with progress
    for task_id, title, _ in tasks:
        mirror_print(f"\n[cyan]Executing:[/cyan] {title}")
        
        # Progress bar
        for i in track(range(10), description=f"Processing {task_id}..."):
            time.sleep(0.2)
            
            # Update console
            console_buffer.truncate(0)
            console_buffer.seek(0)
            console.clear()
            console.print(table)
            console.print(f"\n[cyan]Executing:[/cyan] {title}")
            console.print(f"Progress: [green]{'█' * (i+1)}{'░' * (9-i)}[/green] {(i+1)*10}%")
            broadcast_console()
        
        mirror_print(f"[green]✓ Completed:[/green] {title}")
        time.sleep(1)

if __name__ == '__main__':
    print("🚀 Starting Task Master Mirror Mode")
    print("📺 Terminal: Watch this console")
    print("🌐 Web UI: http://localhost:5000")
    print("✨ Output will be mirrored in real-time!\n")
    
    # Start demo in background
    demo_thread = threading.Thread(target=demo_task_execution, daemon=True)
    demo_thread.start()
    
    # Run Flask app
    socketio.run(app, debug=False, port=5000)