#!/bin/bash

# MemEvolve Linux Uninstall Script
# Removes startup scripts and PATH modifications

echo "🗑️  MemEvolve Linux Uninstall"
echo "============================"

# Check if running interactively
if [ ! -t 0 ] || [ ! -t 1 ]; then
    echo "❌ This script requires an interactive terminal."
    echo "Please run it directly: ./scripts/uninstall.sh"
    exit 1
fi

echo "This will remove:"
echo "- MemEvolveAPI.sh from ~/bin"
echo "- PATH modification from ~/.bashrc"
echo "- memevolve alias from ~/.bashrc"
echo ""

if ! prompt_yes_no "Continue with uninstall?"; then
    echo "Uninstall cancelled."
    exit 0
fi

# Function to prompt yes/no
prompt_yes_no() {
    local prompt="$1"
    local default="${2:-n}"

    echo -n "$prompt (y/N): " >&2
    read response
    case "$response" in
        [yY]|[yY][eE][sS]) return 0 ;;
        [nN]|[nN][oO]|"") [[ "$default" == "y" ]] && return 0 || return 1 ;;
        *) echo "Please answer y or n."; prompt_yes_no "$prompt" "$default" ;;
    esac
}

# Remove startup script
SCRIPT_PATH="$HOME/bin/MemEvolveAPI.sh"
if [ -f "$SCRIPT_PATH" ]; then
    rm "$SCRIPT_PATH"
    echo "✅ Removed $SCRIPT_PATH"
else
    echo "⚠️  $SCRIPT_PATH not found"
fi

# Remove PATH modification
if grep -q 'export PATH="$HOME/bin:$PATH"' ~/.bashrc; then
    sed -i '/export PATH="$HOME\/bin:$PATH"/d' ~/.bashrc
    echo "✅ Removed PATH modification from ~/.bashrc"
else
    echo "⚠️  PATH modification not found in ~/.bashrc"
fi

# Remove alias
if grep -q 'alias memevolve=' ~/.bashrc; then
    sed -i '/alias memevolve=/d' ~/.bashrc
    echo "✅ Removed memevolve alias from ~/.bashrc"
else
    echo "⚠️  memevolve alias not found in ~/.bashrc"
fi

# Remove ~/bin if empty
if [ -d "$HOME/bin" ] && [ -z "$(ls -A $HOME/bin)" ]; then
    rmdir "$HOME/bin"
    echo "✅ Removed empty ~/bin directory"
fi

echo ""
echo "🎉 Uninstall complete!"
echo ""
echo "To apply changes immediately:"
echo "  source ~/.bashrc"
echo ""
echo "Or restart your shell."
echo ""
echo "Note: This does not remove the .env file or virtual environment."
echo "You can manually delete them if desired."