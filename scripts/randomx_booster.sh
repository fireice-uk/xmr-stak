#!/usr/bin/env bash
# based on xmrig's randomx_boost.sh script
# lifted by psychocrypt

function help()
{
    echo "$(basename $0) modifies caching behaviors of your CPU"
    echo "and activates huge pages."
    echo "Reboot your system to revert the changes."
    echo ""
    echo "must be called with administrative privileges e.g. 'sudo $(basename $0)'"
}

if [ $# -ge 1 ] ; then
    help
    exit 1
fi

hasAptGet=$(which apt-get >/dev/null && { echo 1; } || { echo 0; })
hasApt=$(which apt >/dev/null && { echo 1; } || { echo 0; })
hasYum=$(which yum >/dev/null && { echo 1; } || { echo 0; })

tools=$(which wrmsr >/dev/null || { echo "msr-tools "; })$(which numactl >/dev/null || { echo " numactl"; })

if [ -n "$tools" ] ; then
    echo "tool '$tools' not found, $(basename $0) is trying to install the dependency"
    if [ $hasAptGet -eq 1 ] ; then
        comm="apt-get --no-install-recommends --yes install $tools"
        echo "execute: $comm"
        $comm
    elif [ $hasApt -eq 1 ] ; then
        comm="apt-get --no-install-recommends --yes install $tools"
        echo "execute: $comm"
        $comm
    elif [ $hasYum -eq 1 ] ; then
        comm="yum install -y $tools"
        echo "execute: $comm"
        $comm
    else
        echo "package manager unknown, please install '$tools' by hand" >&2
        exit 1
    fi
fi

hasWrmsr=$(which wrmsr >/dev/null && { echo 1; } || { echo 0; })
if [ $hasWrmsr -eq 0 ] ; then
    echo "dependency 'wrmsr' not found, tool failed" >&2
exit 1
fi

hasNumactl=$(which numactl >/dev/null && { echo 1; } || { echo 0; })
if [ $hasNumactl -eq 0 ] ; then
    echo "dependency 'numactl' not found, tool failed" >&2
    exit 1
fi

echo ""
modprobe msr

if cat /proc/cpuinfo | grep -q "AMD Ryzen" ; then
    echo "Detected Ryzen"
    wrmsr -a 0xc0011022 0x510000
    wrmsr -a 0xc001102b 0x1808cc16
    wrmsr -a 0xc0011020 0
    echo "MSR register values for Ryzen applied"
    echo "WARNING: MSR register changes can result into stability issues!"
    echo "Reboot your system to revert the changes."
elif cat /proc/cpuinfo | grep -q "Intel" ; then
    echo "Detected Intel"
    wrmsr -a 0x1a4 7
    echo "MSR register values for Intel applied"
    echo "WARNING: MSR register changes can result into stability issues!"
    echo "Reboot your system to revert the changes."
else
    echo "No supported CPU detected"
fi

echo ""

### begin enable huge pages
required_num_huge_pages=1280
num_huge_pages=$(cat /proc/meminfo | grep "HugePages_Free" | sed 's/ \{2,\}/ /g' | cut -d" " -f2)

if [ $num_huge_pages -lt $required_num_huge_pages ] ; then
    echo "active 2 MiB pages"
    echo "execute: sysctl -w vm.nr_hugepages=$required_num_huge_pages"
    sysctl -w vm.nr_hugepages="$required_num_huge_pages"
fi
# verify number of huge pages
num_huge_pages=$(cat /proc/meminfo | grep "HugePages_Free" | sed 's/ \{2,\}/ /g' | cut -d" " -f2)
num_memsets=$((num_huge_pages/required_num_huge_pages))

if [ $num_memsets -eq 0 ] ; then
    echo "Error: not enough 2 MiB pages $num_huge_pages/$required_num_huge_pages" >&2
fi

# apply gigabyte pages last because 2MiB pages will give more performance
numNodes=$(numactl --hardware | grep available | cut -d" " -f2)
freeGigPages=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages)
neededGigPages=$((numNodes * 3))

if [ $freeGigPages -lt $neededGigPages ] ; then
    echo ""
    echo "activate 1 GiB pages"
    comm="echo $neededGigPages > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"
    echo "execute: $comm"
    echo "$neededGigPages" > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
fi
### end enable huge pages

exit 0
