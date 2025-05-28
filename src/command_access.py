import subprocess

def execute_command(command):
    """Executes a shell command and returns the output."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"

def list_directory(path='.'):
    """Lists the contents of a directory."""
    command = f'ls {path}'  # Use 'dir' for Windows
    return execute_command(command)

def change_directory(path):
    """Changes the current working directory."""
    try:
        os.chdir(path)
        return f"Changed directory to {path}"
    except FileNotFoundError:
        return f"Error: Directory {path} not found."
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_directory():
    """Returns the current working directory."""
    return os.getcwd()