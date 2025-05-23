#!/bin/bash
# Setup script to fix idcodes shared library issues for the Ubuntu 23.04 ARM64 container

set -e  # Exit immediately if a command fails

echo "===== Setting up idcodes library ====="

# Install required Boost dependencies for idcodeslibrary
apt-get update
apt-get install -y libboost-all-dev libboost-program-options-dev libboost-program-options1.83.0


# Create directory for idcodes libraries if it doesn't exist
mkdir -p /usr/local/lib/idcodes

# Install the Python wheel for idcodes (x86_64 version)
echo "Installing idcodes x86_64 wheel..."
pip install --break-system-packages idcodes-0.2.7-cp312-cp312-linux_x86_64.whl

# Install the .deb package for idcodeslibrary (x86_64 version)
echo "Installing idcodeslibrary .deb package..."
dpkg -i idcodeslibrary-0.2.7-x86_64-Ubuntu-24.04.deb || apt-get install -f -y

# Find the installed module location
SITE_PACKAGES_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
IDCODES_DIR="${SITE_PACKAGES_DIR}/idcodes"
echo "Idcodes library installed at: ${IDCODES_DIR}"

# Create symlinks for the shared libraries
if [ -f "${IDCODES_DIR}/idcodes.cpython-312-x86_64-linux-gnu.so" ]; then
    echo "Creating required symlinks..."
    # Create symlinks for libIdcodesLibrary.so.0
    ln -sf "${IDCODES_DIR}/idcodes.cpython-312-x86_64-linux-gnu.so" /usr/local/lib/libIdcodesLibrary.so.0
    ln -sf "${IDCODES_DIR}/idcodes.cpython-312-x86_64-linux-gnu.so" /usr/local/lib/idcodes/libIdcodesLibrary.so.0
    
    # Update system library cache
    ldconfig
    
    # Check for the shared library in the system's library cache
    echo "Checking library cache:"
    ldconfig -p | grep -E 'IdcodesLibrary|crypto'
    
    # Verify libstdc++ version
    echo "Checking libstdc++ version:"
    strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep 'GLIBCXX_3.4.32'
    
    # Create a custom file for environment configuration
    echo "export LD_LIBRARY_PATH=${IDCODES_DIR}:/usr/local/lib/idcodes:/usr/local/lib:${LD_LIBRARY_PATH}" > /etc/profile.d/idcodes.sh
    chmod +x /etc/profile.d/idcodes.sh
    
    # Also add to current session
    export LD_LIBRARY_PATH="${IDCODES_DIR}:/usr/local/lib/idcodes:/usr/local/lib:${LD_LIBRARY_PATH}"
    
    echo "===== Setup completed successfully ====="
    echo "The idcodes library has been configured."
    echo "Library path set to: $LD_LIBRARY_PATH"
else
    echo "Error: idcodes shared object not found at ${IDCODES_DIR}"
    echo "Installation failed."
    exit 1
fi
