# Setup Nahual servers

# ViT-based models
for i in {0..36}; do
	for model in morphem openphenom; do
		session_name="${model}_${i}"
		ipc_addr="ipc:///tmp/${session_name}.ipc"
		echo "Starting ${model} instance in screen session '${session_name}'"
		screen -S "${session_name}" -dm bash -c "nix run github:afermg/nahual_vit#${model} '${ipc_addr}'"
	done
done

# Subcell
for i in {0..9}; do
	session_name="subcell_${i}"
	ipc_addr="ipc:///tmp/subcell_${i}.ipc"
	echo "Starting subcell instance in screen session '${session_name}'"
	screen -S "${session_name}" -dm bash -c "nix run github:afermg/SubCellPortable '${ipc_addr}'"
done

echo "All instances started in detached screen sessions."
echo "Use 'screen -ls' to list sessions and 'screen -r \$MODEL_\$InstanceID' to attach one in partiular."
echo "To kill them all, run: screen -ls | awk -F'.' '/\S+_[0-9]/ {print $1}' | xargs kill"

# Run aliby that uses servers for embeddings
# python aliby_featurize.py
