# Set GIT_SSH_COMMAND if not already set, so git uses the workspace SSH key
if [[ -z "$GIT_SSH_COMMAND" ]]; then
  echo 'export GIT_SSH_COMMAND="ssh -i /workspace/.ssh/id_ed25519 -o IdentitiesOnly=yes"' >> ~/.bashrc
  source ~/.bashrc
fi
git config --global user.name "Tao Luo"
git config --global user.email "taoluo321@outlook.com"

git clone git@github.com:taoluo/_RLix.git RLix
cd RLix
git submodule update --remote --init --checkout external/ROLL_rlix
# git submodule update --remote --init --checkout external/ROLL_upstream_main
# git submodule update --remote --init --checkout external/ROLL_multi_lora
# git submodule update --remote --init --checkout external/ROLL_multi_pipeline

    