#!/bin/bash
# Permanent Battery Backup Optimization Script for Dell G15 5530 (Intel i5 13th Gen + NVIDIA)

echo "=========================================================="
echo " Starting Safe & Permanent Battery Optimization Setup... "
echo "=========================================================="

if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run this script with sudo: sudo bash setup_battery_fix.sh"
  exit 1
fi

# Create a backup of GRUB config for safety
echo "📦 Backing up /etc/default/grub to /etc/default/grub.bak..."
cp /etc/default/grub /etc/default/grub.bak

echo "1️⃣ Removing CPU C-State limit (intel_idle.max_cstate=1) from GRUB..."
sed -i 's/intel_idle.max_cstate=1 //g' /etc/default/grub
update-grub

echo "2️⃣ Configuring NVIDIA PCIe Runtime Power Management (D3cold sleep)..."
cat << 'EOF' > /etc/udev/rules.d/80-nvidia-pm.rules
# Enable runtime power management for NVIDIA graphics and audio devices
ACTION=="add", SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{class}=="0x030000", ATTR{power/control}="auto"
ACTION=="add", SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{class}=="0x030200", ATTR{power/control}="auto"
ACTION=="add", SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{class}=="0x040300", ATTR{power/control}="auto"
EOF

cat << 'EOF' > /etc/modprobe.d/nvidia-pm.conf
options nvidia NVreg_DynamicPowerManagement=0x02
EOF

echo "3️⃣ Setting TLP battery optimizations..."
sed -i 's/^#*CPU_BOOST_ON_BAT=.*/CPU_BOOST_ON_BAT=0/' /etc/tlp.conf
sed -i 's/^#*CPU_ENERGY_PERF_POLICY_ON_BAT=.*/CPU_ENERGY_PERF_POLICY_ON_BAT=power/' /etc/tlp.conf
sed -i 's/^#*PCIE_ASPM_ON_BAT=.*/PCIE_ASPM_ON_BAT=powersave/' /etc/tlp.conf
sed -i 's/^#*WIFI_PWR_ON_BAT=.*/WIFI_PWR_ON_BAT=on/' /etc/tlp.conf

systemctl restart tlp 2>/dev/null || tlp start

echo "=========================================================="
echo " ✅ Battery Backup Optimization Complete!"
echo " ⚠️  PLEASE REBOOT YOUR LAPTOP TO APPLY C-STATE & GPU CHANGES:"
echo "     sudo reboot"
echo "=========================================================="
