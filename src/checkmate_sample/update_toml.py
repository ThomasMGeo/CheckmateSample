import toml
from datetime import datetime

def update_version():
    # Read the current pyproject.toml
    with open('pyproject.toml', 'r') as f:
        config = toml.load(f)
    
    # Update the version with today's date
    today = datetime.today()
    new_version = today.strftime("%Y.%m.%d")
    config['project']['version'] = new_version
    
    # Write the updated config back to pyproject.toml
    with open('pyproject.toml', 'w') as f:
        toml.dump(config, f)

    print(f"Updated version to {new_version}")

if __name__ == "__main__":
    update_version()
