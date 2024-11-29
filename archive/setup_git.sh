# setup_git.sh - Ubuntu
sudo apt update
sudo apt install git
git config --global user.name "Adge Denkers"
git config --global user.email "adge.denkers@gmail.com"
git config --global core.editor "nano"
git config --global color.ui true
git config --global push.default simple

# Generate SSH key
ssh-keygen -t rsa -b 4096 -C "adge.denkers@gmail.com"

# Start the ssh-agent in the background
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa_ubuntu

# Copy the SSH key to the clipboard
sudo apt install xclip
xclip -sel clip < ~/.ssh/id_rsa_ubuntu.pub

# Add the SSH key to GitHub
echo "SSH key copied to clipboard. Add it to GitHub and press any key to continue."
read -n 1 -s

# Test the SSH connection
ssh -T git@github.com
