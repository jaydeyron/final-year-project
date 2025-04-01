from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time
import os

# Create progress file path
progress_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress.json')

class ProgressServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/progress'):
            try:
                # Read progress file
                if os.path.exists(progress_file):
                    with open(progress_file, 'r') as f:
                        data = json.load(f)
                else:
                    data = {"progress": 0}
                
                # Send response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
                
                print(f"Served progress: {data}")
            except Exception as e:
                self.send_error(500, f"Error: {str(e)}")
                print(f"Error serving progress: {e}")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress default logging
        return

def run():
    # Initialize progress file
    with open(progress_file, 'w') as f:
        json.dump({"progress": 0}, f)
    
    # Start server
    server_address = ('', 8888)
    httpd = HTTPServer(server_address, ProgressServer)
    print(f"Starting test progress server at http://localhost:8888/progress")
    
    # Increment progress in the background
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Server stopped.")

if __name__ == "__main__":
    import threading
    
    def update_progress():
        for i in range(0, 101, 5):
            with open(progress_file, 'w') as f:
                json.dump({"progress": i}, f)
            print(f"Updated progress to {i}%")
            time.sleep(1)
    
    # Start progress updater thread
    updater = threading.Thread(target=update_progress)
    updater.daemon = True
    updater.start()
    
    # Run server
    run()
