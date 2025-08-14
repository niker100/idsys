# Dev Container Quickstart in VS Code

## Prerequisites

- [Visual Studio Code](https://code.visualstudio.com/) installed
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed
- [Docker](https://www.docker.com/products/docker-desktop/) running
    - On Windows, be sure that Docker Desktop and so the Docker Engine is started

## Steps

1. **Open Your Project Folder in VS Code**
    - Use `File` > `Open Folder...` and select your project.

2. **Reopen in Container**
    - After configuration, VS Code will prompt: _"Reopen in Container"_. Click it.
    - Alternatively Press `F1` (or `Ctrl+Shift+P`) to use the Command Palette:  
      `Dev Containers: Reopen in Container`

3. **Wait for Build and Attach**
    - VS Code will build the container and attach to it.
    - Your workspace is now running inside the dev container.

4. **(Reopen the Folder locally)**
    - For disconnecting from the container use the Command Palette:
    `Dev Containers: Reopen Folder Locally`

## Resources

- [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers)
- [Dev Containers Reference](https://containers.dev/)
