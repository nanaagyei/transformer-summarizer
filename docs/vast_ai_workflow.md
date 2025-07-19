# Vast.ai Correct Workflow

## 🚨 The Issue

The previous approach was **incorrect**. The error you encountered:

```bash
vastai create instance --from-json 'vast_job.json'
usage: vastai create instance ID [OPTIONS] [--args ...]
vastai create instance: error: argument id: invalid int value: 'vast_job.json'
```

This happened because the `vastai create instance` command expects an **instance ID** as the first argument, not a JSON file.

## ✅ Correct Vast.ai Workflow

Based on the [official Vast.ai API documentation](https://docs.vast.ai/api/search-templates), the correct workflow is:

### Step 1: Search for Available Instances

```bash
# Search for RTX 4090 instances
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100'

# Search for cost-effective options
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100 price_per_hour<0.50'
```

This returns a list of available instances with their IDs.

### Step 2: Create Instance Using Instance ID

```bash
# Use the instance ID from the search results
vastai create instance <instance_id> --ssh-key <your_ssh_key> --onstart <startup_script>
```

**NOT** `vastai create instance --from-json vast_job.json`

## 🔧 Our Automated Solution

We've created `scripts/deploy.py` to automate this workflow:

```bash
# 1. Search and select instance
python scripts/deploy.py --action search

# 2. Create instance with selected ID
python scripts/deploy.py --action create --instance-id <selected_instance_id>

# 3. SSH into instance
python scripts/deploy.py --action ssh --instance-id <instance_id>

# 4. Monitor instance
python scripts/deploy.py --action monitor --instance-id <instance_id>

# 5. Destroy when done
python scripts/deploy.py --action destroy --instance-id <instance_id>
```

## 📋 What the Script Does

### Search Action

1. Runs `vastai search offers` with your criteria
2. Displays results in a user-friendly table
3. Lets you select an instance
4. Provides the command to create it

### Create Action

1. Takes the instance ID you selected
2. Runs `vastai create instance <id>` with your configuration
3. Applies SSH key and startup script from `vast_job.json`
4. Reports success/failure

### Other Actions

- **list**: Shows all your instances
- **ssh**: SSH into an instance
- **monitor**: Show instance details
- **destroy**: Destroy an instance

## 🎯 Key Differences

| ❌ Wrong Approach                                  | ✅ Correct Approach                    |
| -------------------------------------------------- | -------------------------------------- |
| `vastai create instance --from-json vast_job.json` | `vastai create instance <instance_id>` |
| Assumes JSON file contains instance ID             | Searches for instances first           |
| Doesn't work with Vast.ai API                      | Follows official documentation         |
| No instance selection                              | User selects best instance             |

## 🔍 Search Templates

Based on the [official documentation](https://docs.vast.ai/api/search-templates), effective search queries:

```bash
# Basic search
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100'

# Cost-optimized
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100 price_per_hour<0.50'

# High-performance
vastai search offers 'gpu_name:A100 num_gpus:1 disk_space:>100'

# Spot instances
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100 spot:true'

# Verified hosts
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100 verified:true'
```

## 🚀 Quick Start

1. **Setup your environment:**

   ```bash
   pip install vast-ai
   export VAST_API_KEY=your_api_key_here
   ```

2. **Search for instances:**

   ```bash
   python scripts/deploy.py --action search
   ```

3. **Create instance:**

   ```bash
   python scripts/deploy.py --action create --instance-id <selected_id>
   ```

4. **SSH and monitor:**

   ```bash
   python scripts/deploy.py --action ssh --instance-id <instance_id>
   ```

5. **Clean up:**
   ```bash
   python scripts/deploy.py --action destroy --instance-id <instance_id>
   ```

## 📚 References

- [Official Vast.ai API Documentation](https://docs.vast.ai/api/search-templates)
- [Vast.ai CLI Documentation](https://docs.vast.ai/cli/)
- [Vast.ai PyTorch Guide](https://docs.vast.ai/pytorch)

## 🐛 Troubleshooting

If you still encounter issues:

1. **Check API key:**

   ```bash
   echo $VAST_API_KEY
   ```

2. **Test CLI:**

   ```bash
   vastai show credit
   ```

3. **Check instance availability:**

   ```bash
   vastai search offers 'gpu_name:RTX_4090' --raw | head -20
   ```

4. **Verify SSH key:**
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```

The key insight is that Vast.ai requires you to **search first, then create** - you can't create an instance without knowing which specific instance you want to use.
