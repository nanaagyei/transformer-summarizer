# Complete Vast.ai Deployment Guide

## ✅ Problem Solved!

The original issue was using incorrect Vast.ai command syntax. This guide provides the complete working solution.

## 🚨 The Original Problem

```bash
# ❌ WRONG - This doesn't work
vastai create instance --from-json vast_job.json
```

**Error:** `argument id: invalid int value: 'vast_job.json'`

## ✅ The Correct Solution

The correct Vast.ai workflow is:

1. **Search for available instances** first
2. **Create instance using the selected instance ID**
3. **SSH into the instance** using the provided SSH details

## 🚀 Complete Working Workflow

### Step 1: Search and Select Instance

```bash
python scripts/deploy.py --action search
```

This will:

- Search for RTX 4090 instances with your criteria
- Display results in a user-friendly table
- Let you select an instance
- Provide the command to create it

**Example Output:**

```
🔍 Searching for RTX_4090 instances with max price $0.5/hr...
Running: vastai search offers gpu_name=RTX_4090 num_gpus=1 disk_space>100 dph_total<0.5 --raw
✅ Found 51 matching instances:

 1. ID: 21614864 | GPU: RTX 4090     | Price: $0.254/hr | Disk: 253GB | Reliability: 0.994
 2. ID: 19115553 | GPU: RTX 4090     | Price: $0.241/hr | Disk: 753GB | Reliability: 0.996
 3. ID: 19923959 | GPU: RTX 4090     | Price: $0.362/hr | Disk: 1020GB | Reliability: 0.997
...

Select instance (1-10) or 'q' to quit: 1

✅ Selected instance 21614864
   GPU: RTX 4090
   Price: $0.254/hr
   Disk: 253GB

🚀 To create this instance, run:
python scripts/deploy.py --action create --instance-id 21614864
```

### Step 2: Create Instance

```bash
python scripts/deploy.py --action create --instance-id <selected_instance_id>
```

This will:

- Create the instance using the correct Vast.ai syntax
- Apply your startup script and configuration
- Return the new instance ID

**Example Output:**

```
🚀 Creating instance 21614864...
Running command: vastai create instance 21614864 --image...
✅ Instance created successfully!
Output: {'success': True, 'new_contract': 23502468}

📋 New instance ID: 23502468

✅ Instance created! To SSH into it, run:
python scripts/deploy.py --action ssh --instance-id 23502468
```

### Step 3: SSH into Instance

```bash
python scripts/deploy.py --action ssh --instance-id <instance_id>
```

**Note:** You need the private key file (`vastai.pem`) to SSH into the instance.

**Example Output:**

```
🔗 SSH-ing into instance 23502468...
📡 SSH URL: ssh://root@ssh2.vast.ai:22468
⚠️  Private key file 'vastai.pem' not found!
💡 You need the private key file to SSH into the instance.
   The public key file 'vastai.pub' is not sufficient for SSH access.

🔧 To SSH manually once you have the private key:
   ssh -i vastai.pem -p 22468 root@ssh2.vast.ai

📋 SSH Details:
   Host: ssh2.vast.ai
   Port: 22468
   User: root
   Key: vastai.pem (private key file)
```

### Step 4: Monitor Instance

```bash
python scripts/deploy.py --action monitor --instance-id <instance_id>
```

**Example Output:**

```
📊 Monitoring instance 23502468...
ID        Machine  Status   Num  Model     Util. %  vCPUs    RAM  Storage  SSH Addr      SSH Port  $/hr    Image
23502468  11230    running   1x  RTX_4090  0.0      21.3   257.7  100      ssh2.vast.ai  22468     0.2944  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```

### Step 5: List All Instances

```bash
python scripts/deploy.py --action list
```

### Step 6: Destroy Instance (when done)

```bash
python scripts/deploy.py --action destroy --instance-id <instance_id>
```

## 🔧 Key Fixes Made

### 1. Search Query Syntax

- **Before:** `gpu_name:RTX_4090` ❌
- **After:** `gpu_name=RTX_4090` ✅

### 2. Price Field

- **Before:** `price_per_hour` ❌
- **After:** `dph_total` ✅

### 3. Create Command

- **Before:** `vastai create instance --from-json vast_job.json` ❌
- **After:** `vastai create instance <id> --image <image> --ssh --direct --onstart-cmd <script>` ✅

### 4. SSH Command

- **Before:** `vastai ssh <id>` ❌
- **After:** `ssh -i vastai.pem -p <port> root@<host>` ✅

### 5. Startup Script

- **Before:** `--onstart` ❌
- **After:** `--onstart-cmd` ✅

## 📋 Required Files

### 1. SSH Key Files

You need both the public and private key files:

```bash
# Public key (you have this)
vastai.pub

# Private key (you need this for SSH access)
vastai.pem
```

### 2. Configuration Files

```bash
# Vast.ai job configuration
vast_job.json

# Deployment script
scripts/deploy.py
```

## 🔑 SSH Key Setup

To SSH into your instances, you need the private key file. The public key file (`vastai.pub`) is not sufficient.

### Option 1: Generate New SSH Key Pair

```bash
# Generate new SSH key pair
ssh-keygen -t rsa -b 4096 -f vastai -C "your_email@example.com"

# This creates:
# - vastai (private key)
# - vastai.pub (public key)

# Add public key to Vast.ai
cat vastai.pub | vastai create ssh-key

# Use private key for SSH
ssh -i vastai -p 22468 root@ssh2.vast.ai
```

### Option 2: Use Existing SSH Key

If you have an existing SSH key pair, copy the private key to `vastai.pem`:

```bash
cp ~/.ssh/id_rsa vastai.pem
chmod 600 vastai.pem
```

## 🚀 Quick Start Commands

```bash
# 1. Search and select instance
python scripts/deploy.py --action search

# 2. Create instance (after selecting from search)
python scripts/deploy.py --action create --instance-id <selected_id>

# 3. List your instances
python scripts/deploy.py --action list

# 4. Monitor instance
python scripts/deploy.py --action monitor --instance-id <instance_id>

# 5. SSH into instance (requires private key)
python scripts/deploy.py --action ssh --instance-id <instance_id>

# 6. Destroy when done
python scripts/deploy.py --action destroy --instance-id <instance_id>
```

## 🐛 Troubleshooting

### Account Verification

If you get "Your account is not verified!":

1. Go to https://vast.ai/console/account
2. Verify your email address
3. Add credits to your account

### SSH Connection Issues

If SSH fails with "Permission denied (publickey)":

1. Make sure you have the private key file (`vastai.pem`)
2. Check that the private key matches the public key in your Vast.ai account
3. Try SSH-ing manually: `ssh -i vastai.pem -p <port> root@<host>`

### Instance Creation Fails

If instance creation fails:

1. Check your credit balance: `vastai show credit`
2. Verify your API key: `echo $VAST_API_KEY`
3. Check instance availability: `vastai search offers 'gpu_name=RTX_4090'`

## 📚 References

- [Official Vast.ai API Documentation](https://docs.vast.ai/api/search-templates)
- [Vast.ai CLI Documentation](https://docs.vast.ai/cli/)
- [Vast.ai PyTorch Guide](https://docs.vast.ai/pytorch)

## 🎯 Summary

The deployment script now correctly:

- ✅ Searches for instances using proper Vast.ai syntax
- ✅ Creates instances with the correct command format
- ✅ Provides SSH details and instructions
- ✅ Monitors and manages instances
- ✅ Follows official Vast.ai documentation

The key insight is that Vast.ai requires you to **search first, then create** - you can't create an instance without knowing which specific instance you want to use.
