#!/usr/bin/env python3
"""
Vast.ai Deployment Script

This script follows the correct Vast.ai workflow:
1. Search for available instances
2. Create instance using selected instance ID
3. Monitor and manage the instance
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


class VastAIDeployer:
    """Deploy and manage Vast.ai instances following official documentation"""

    def __init__(self, config_path: str = "vast_job.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load Vast.ai job configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def search_instances(self, gpu_type: str = "RTX_4090", max_price: float = 0.5,
                         disk_space: int = 100, num_gpus: int = 1) -> List[Dict[str, Any]]:
        """Search for available instances matching criteria"""
        print(
            f"🔍 Searching for {gpu_type} instances with max price ${max_price}/hr...")

        # Build search query based on Vast.ai documentation
        # Note: Use = instead of : for field comparisons, and > instead of > for numeric comparisons
        search_query = f"gpu_name={gpu_type} num_gpus={num_gpus} disk_space>{disk_space} dph_total<{max_price}"

        try:
            # Run vastai search command
            cmd = ["vastai", "search", "offers", search_query, "--raw"]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)

            # Check if output is empty
            if not result.stdout.strip():
                print("❌ No instances found matching your criteria")
                return []

            # Parse the JSON output
            offers = json.loads(result.stdout)

            if not offers:
                print("❌ No instances found matching your criteria")
                return []

            print(f"✅ Found {len(offers)} matching instances:")
            print()

            # Display instances in a table format
            for i, offer in enumerate(offers[:10]):  # Show first 10
                gpu_name = offer.get('gpu_name', 'Unknown')
                price = offer.get('dph_total', 0)
                disk = offer.get('disk_space', 0)
                instance_id = offer.get('id', 'Unknown')
                reliability = offer.get('reliability', 0)

                print(f"{i+1:2d}. ID: {str(instance_id):8s} | GPU: {gpu_name:12s} | "
                      f"Price: ${price:.3f}/hr | Disk: {disk:.0f}GB | Reliability: {reliability:.3f}")

            if len(offers) > 10:
                print(f"   ... and {len(offers) - 10} more instances")

            return offers

        except subprocess.CalledProcessError as e:
            print(f"❌ Error searching instances: {e}")
            print(f"Command output: {e.stderr}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing search results: {e}")
            # Show first 200 chars
            print(f"Raw output: {result.stdout[:200]}...")
            return []

    def select_instance(self, offers: List[Dict[str, Any]]) -> Optional[str]:
        """Let user select an instance from search results"""
        if not offers:
            return None

        while True:
            try:
                choice = input(
                    f"\nSelect instance (1-{min(len(offers), 10)}) or 'q' to quit: ").strip()

                if choice.lower() == 'q':
                    return None

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(offers):
                    selected_offer = offers[choice_idx]
                    instance_id = selected_offer.get('id')

                    print(f"\n✅ Selected instance {instance_id}")
                    print(f"   GPU: {selected_offer.get('gpu_name')}")
                    print(
                        f"   Price: ${selected_offer.get('dph_total', 0):.3f}/hr")
                    print(f"   Disk: {selected_offer.get('disk_space')}GB")

                    return str(instance_id)
                else:
                    print("❌ Invalid selection. Please try again.")

            except ValueError:
                print("❌ Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n❌ Cancelled by user.")
                return None

    def create_instance(self, instance_id: str) -> bool:
        """Create instance using the selected instance ID"""
        print(f"🚀 Creating instance {instance_id}...")

        try:
            # Use the correct vastai create instance command
            cmd = ["vastai", "create", "instance", instance_id]

            # Add required image parameter
            if self.config.get('image'):
                cmd.extend(["--image", self.config['image']])

            # Add configuration options from our JSON
            if self.config.get('startup_script'):
                cmd.extend(["--onstart-cmd", self.config['startup_script']])

            if self.config.get('disk_space'):
                cmd.extend(["--disk", str(self.config['disk_space'])])

            # Add SSH support (uses default SSH key from Vast.ai account)
            cmd.extend(["--ssh", "--direct"])

            # Don't show the full startup script
            print(f"Running command: {' '.join(cmd[:5])}...")

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)

            print("✅ Instance created successfully!")
            print(f"Output: {result.stdout}")

            # Parse the JSON response
            try:
                response = json.loads(result.stdout)
                if response.get('success') and response.get('new_contract'):
                    new_instance_id = response['new_contract']
                    print(f"📋 New instance ID: {new_instance_id}")
                    return True
            except json.JSONDecodeError:
                pass

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating instance: {e}")
            print(f"Command output: {e.stderr}")
            return False

    def list_instances(self):
        """List all user instances"""
        print("📋 Listing your instances...")

        try:
            cmd = ["vastai", "show", "instances"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)

            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"❌ Error listing instances: {e}")
            print(f"Command output: {e.stderr}")

    def ssh_to_instance(self, instance_id: str):
        """SSH into an instance"""
        print(f"🔗 SSH-ing into instance {instance_id}...")

        try:
            # Get SSH URL for the instance
            cmd = ["vastai", "ssh-url", instance_id]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            ssh_url = result.stdout.strip()

            print(f"📡 SSH URL: {ssh_url}")

            # Check for private key file
            private_key_file = Path("vastai.pem")
            if private_key_file.exists():
                print("🔑 Found private key file: vastai.pem")
                print("💡 To SSH manually, use:")
                print(f"   ssh -i vastai.pem -p 22468 root@ssh2.vast.ai")
                print()
                print("🔗 Attempting to connect...")

                # Parse SSH URL to extract host and port
                ssh_url_clean = ssh_url.replace("ssh://", "")
                if ":" in ssh_url_clean:
                    host_part, port = ssh_url_clean.split(":")
                    ssh_cmd = ["ssh", "-i",
                               str(private_key_file), "-p", port, host_part]
                else:
                    ssh_cmd = ["ssh", "-i",
                               str(private_key_file), ssh_url_clean]
                subprocess.run(ssh_cmd, check=True)
            else:
                print("⚠️  Private key file 'vastai.pem' not found!")
                print("💡 You need the private key file to SSH into the instance.")
                print(
                    "   The public key file 'vastai.pub' is not sufficient for SSH access.")
                print()
                print("🔧 To SSH manually once you have the private key:")
                print(f"   ssh -i vastai.pem -p 22468 root@ssh2.vast.ai")
                print()
                print("📋 SSH Details:")
                print(f"   Host: ssh2.vast.ai")
                print(f"   Port: 22468")
                print(f"   User: root")
                print(f"   Key: vastai.pem (private key file)")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error SSH-ing to instance: {e}")
            print("💡 You can try SSH-ing manually using the command above.")
        except KeyboardInterrupt:
            print("\n👋 Disconnected from instance.")

    def destroy_instance(self, instance_id: str):
        """Destroy an instance"""
        print(f"🗑️ Destroying instance {instance_id}...")

        try:
            cmd = ["vastai", "destroy", "instance", instance_id]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)

            print("✅ Instance destroyed successfully!")
            print(f"Output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error destroying instance: {e}")
            print(f"Command output: {e.stderr}")

    def monitor_instance(self, instance_id: str):
        """Monitor an instance"""
        print(f"📊 Monitoring instance {instance_id}...")

        try:
            cmd = ["vastai", "show", "instance", instance_id]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)

            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"❌ Error monitoring instance: {e}")
            print(f"Command output: {e.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Vast.ai Instance Deployment Tool")
    parser.add_argument("--config", default="vast_job.json",
                        help="Path to Vast.ai job config")
    parser.add_argument("--action", choices=["search", "create", "list", "ssh", "destroy", "monitor"],
                        default="search", help="Action to perform")
    parser.add_argument(
        "--instance-id", help="Instance ID for SSH/destroy/monitor actions")
    parser.add_argument("--gpu-type", default="RTX_4090",
                        help="GPU type to search for")
    parser.add_argument("--max-price", type=float,
                        default=0.5, help="Maximum price per hour")
    parser.add_argument("--disk-space", type=int, default=100,
                        help="Minimum disk space in GB")
    parser.add_argument("--num-gpus", type=int,
                        default=1, help="Number of GPUs")

    args = parser.parse_args()

    try:
        deployer = VastAIDeployer(args.config)

        if args.action == "search":
            offers = deployer.search_instances(
                gpu_type=args.gpu_type,
                max_price=args.max_price,
                disk_space=args.disk_space,
                num_gpus=args.num_gpus
            )

            if offers:
                instance_id = deployer.select_instance(offers)
                if instance_id:
                    print(f"\n🚀 To create this instance, run:")
                    print(
                        f"python scripts/deploy.py --action create --instance-id {instance_id}")

        elif args.action == "create":
            if not args.instance_id:
                print("❌ Please provide an instance ID with --instance-id")
                sys.exit(1)

            success = deployer.create_instance(args.instance_id)
            if success:
                print(f"\n✅ Instance created! To SSH into it, run:")
                print(
                    f"python scripts/deploy.py --action ssh --instance-id {args.instance_id}")

        elif args.action == "list":
            deployer.list_instances()

        elif args.action == "ssh":
            if not args.instance_id:
                print("❌ Please provide an instance ID with --instance-id")
                sys.exit(1)
            deployer.ssh_to_instance(args.instance_id)

        elif args.action == "destroy":
            if not args.instance_id:
                print("❌ Please provide an instance ID with --instance-id")
                sys.exit(1)
            deployer.destroy_instance(args.instance_id)

        elif args.action == "monitor":
            if not args.instance_id:
                print("❌ Please provide an instance ID with --instance-id")
                sys.exit(1)
            deployer.monitor_instance(args.instance_id)

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Make sure you have run the cloud training setup first:")
        print("   python scripts/cloud_train.py --provider vast")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
