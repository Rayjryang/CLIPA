log_name="your_log_name.log"
out_dir='./train_logs'

# Check if the output directory exists, create it if not
if [ ! -d "$out_dir" ]; then
    mkdir -p "$out_dir"
fi

nohup bash pre_training.sh > "$out_dir/$log_name" 2>&1 &
