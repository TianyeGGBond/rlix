touch ~/.no_auto_tmux

# Set GIT_SSH_COMMAND if not already set, so git uses the workspace SSH key
if [[ -z "$GIT_SSH_COMMAND" ]]; then
  echo 'export GIT_SSH_COMMAND="ssh -i /workspace/.ssh/id_ed25519 -o IdentitiesOnly=yes"' >> ~/.bashrc
  source ~/.bashrc

fi

git config --global user.name "Tao Luo"
git config --global user.email "taoluo321@outlook.com"

# export RAY_grpc_server_thread_pool_size=4 # reduce the thread usage to save limited pid resouce 
cd external/ROLL_schedrl

conda init && conda activate main # ensure we are in main env 
# Warn if python is not 3.10, but do not exit
python_version=$(python --version 2>&1 | grep -oP '\d+\.\d+')
if [[ "$python_version" != "3.10"* ]]; then
  echo "ERROR: expected Python 3.10, got $python_version, use cuda 12.4 docker container"
  exit 1
fi


uv pip install -r requirements_torch260_vllm.txt
uv pip install --no-build-isolation transformer-engine[pytorch]==2.2.0

# for tracing
apt-get update && apt-get install -y protobuf-compiler libprotobuf-dev
uv pip install "protobuf<3.21.0" "tg4perfetto>=0.0.6"



# curl -fsSL https://opencode.ai/install | bash
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
npm i -g @openai/codex
bash -c "$(curl -fsSL https://gitee.com/iflow-ai/iflow-cli/raw/main/install.sh)"
