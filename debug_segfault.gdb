# GDB script for debugging segmentation fault in GausStudio
# Usage: gdb -x debug_segfault.gdb ./GausStudio

# Set up useful commands
set pagination off
set confirm off

# Enable core dumps
shell ulimit -c unlimited

# Set breakpoints at key functions
break DicomReader::readDirectory
break DicomReader::loadDataMultithreaded
break DicomReader::loadData_thread
break Interface::ShowViewerWindow
break UpdateTextures

# Set breakpoint on segmentation fault
catch signal SIGSEGV

# Commands to run when program starts
define startup
    echo Starting GausStudio with debug info...\n
    run
end

# Commands to run when segmentation fault occurs
define segfault_handler
    echo \n=== SEGMENTATION FAULT DETECTED ===\n
    echo Backtrace:\n
    bt
    echo \n=== DETAILED BACKTRACE ===\n
    bt full
    echo \n=== REGISTERS ===\n
    info registers
    echo \n=== LOCAL VARIABLES ===\n
    info locals
    echo \n=== ARGUMENTS ===\n
    info args
    echo \n=== THREAD INFO ===\n
    info threads
    echo \n=== MEMORY AROUND CRASH ===\n
    x/20i $pc
end

# Set up the segfault handler
commands
    segfault_handler
    continue
end

# Start the program
startup 