#include "vrpn_Tracker.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <csignal>
#include <fstream>
#include <thread>

/*
 * WorldViz Position Tracker Logger
 *
 * This program connects to a VRPN (Virtual Reality Peripheral Network) tracker
 * server and logs ground-truth UE positions to a text file. The logged positions
 * are used as ground-truth labels for supervised neural positioning training.
 *
 * Functionality:
 * - Connects to a VRPN tracker server (default: localhost:3883)
 * - Receives position and orientation updates from the tracker
 * - Logs timestamps and positions to "position_data.txt"
 * - Handles graceful shutdown on SIGINT (Ctrl+C)
 *
 * Usage:
 *   ./tracker_logging_worldviz [hostname] [port]
 *
 *   Arguments:
 *     hostname: VRPN tracker server hostname (default: "localhost")
 *     port:     VRPN tracker server port (default: 3883)
 *
 * Output Format:
 *   Each line in position_data.txt contains:
 *   YYYY-MM-DD HH:MM:SS.microseconds x,y,z;q0,q1,q2,q3,
 *   where (x,y,z) are position coordinates and (q0,q1,q2,q3) are quaternion
 *   orientation values.
 *
 * Requirements:
 *   - VRPN library: https://github.com/vrpn/vrpn
 *   - Compile within the VRPN codebase
 *
 * Created on Fri Feb 23 13:21:40 2024
 * @author: Reinhard Wiesmayr
 */


using namespace std;

///////////////////////////////////////////////////////////////////////////////
// Global Variables
///////////////////////////////////////////////////////////////////////////////

// Global file stream for position logging
std::ofstream outfile;

// Flag to indicate program termination (set by signal handler)
volatile sig_atomic_t exit_flag = 0;

///////////////////////////////////////////////////////////////////////////////
// Signal Handler
///////////////////////////////////////////////////////////////////////////////

/**
 * Signal handler for SIGINT (Ctrl+C).
 * Gracefully flushes and closes the output file before exiting.
 */
void handle_signal(int signal) {
    std::cout << "\nCaught signal " << signal << ", flushing file..." << std::endl;

    // Set the termination flag to break out of the loop
    exit_flag = 1;

    // Flush and close the file
    outfile.flush();
    outfile.close();
    
    std::cout << "File flushed and closed. Exiting program." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
// VRPN Tracker Callback
///////////////////////////////////////////////////////////////////////////////

/**
 * Callback function called by VRPN when a tracker update is received.
 * Formats and logs the position and orientation data with a timestamp.
 *
 * @param userData: User data pointer (unused)
 * @param t: VRPN tracker callback data containing position and quaternion
 */
void VRPN_CALLBACK handle_tracker(void* userData, const vrpn_TRACKERCB t) {
    // Get current system time
    auto now = std::chrono::system_clock::now();

    // Extract time since epoch in seconds and microseconds
    auto duration = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration) - seconds;

    // Convert to system clock time for formatting
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm utc_tm = *std::gmtime(&now_time);

    // Format: YYYY-MM-DD HH:MM:SS.microseconds x,y,z;q0,q1,q2,q3,
    // where (x,y,z) is position and (q0,q1,q2,q3) is quaternion orientation
    // Print to console
    cout << std::put_time(&utc_tm, "%Y-%m-%d %H:%M:%S")
              << "." << std::setfill('0') << setw(6) << microseconds.count()
              << " " << t.pos[0] << "," << t.pos[1] << "," << t.pos[2] 
              << ";" << t.quat[0] << "," << t.quat[1] << "," << t.quat[2] << "," << t.quat[3] << "," << endl;

    // Write to file
    outfile << std::put_time(&utc_tm, "%Y-%m-%d %H:%M:%S")
            << "." << std::setfill('0') << setw(6) << microseconds.count()
            << " " << t.pos[0] << "," << t.pos[1] << "," << t.pos[2] 
            << ";" << t.quat[0] << "," << t.quat[1] << "," << t.quat[2] << "," << t.quat[3] << "," << endl;
}

///////////////////////////////////////////////////////////////////////////////
// Main Function
///////////////////////////////////////////////////////////////////////////////

/**
 * Main function: Sets up VRPN tracker connection and logging loop.
 * 
 * Command-line arguments:
 *   argv[1]: VRPN tracker server hostname (optional, default: "localhost")
 *   argv[2]: VRPN tracker server port (optional, default: 3883)
 */
int main(int argc, char* argv[]) {


    // Parse command-line arguments
    std::string hostname = (argc > 1) ? argv[1] : "localhost";
    int port = (argc > 2) ? std::stoi(argv[2]) : 3883;

    // Construct VRPN connection string (format: "PPT0@hostname:port")
    std::string connection_name = "PPT0@" + hostname + ":" + std::to_string(port);
    char connection_name_arr[connection_name.length() + 1];
    strcpy(connection_name_arr, connection_name.c_str());

    // Create VRPN tracker remote connection
    std::cout << "Connecting to VRPN tracker: " << connection_name << std::endl;
    vrpn_Tracker_Remote* vrpnTracker = new vrpn_Tracker_Remote(connection_name_arr);
    
    // Register callback function for tracker updates
    vrpnTracker->register_change_handler(0, handle_tracker, 0);

    // Open output file in append mode (preserves existing data)
    outfile.open("position_data.txt", std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open position_data.txt for writing." << std::endl;
        return 1;
    }
    std::cout << "Logging positions to position_data.txt" << std::endl;

    // Register signal handler for graceful shutdown (Ctrl+C)
    signal(SIGINT, handle_signal);

    // Main loop: process VRPN messages until termination signal
    std::cout << "Tracking started. Press Ctrl+C to stop and save." << std::endl;
    while (!exit_flag) {
        vrpnTracker->mainloop();  // Process incoming VRPN messages
        
        // Small sleep to prevent excessive CPU usage
        vrpn_SleepMsecs(1);
    }

    // Cleanup
    delete vrpnTracker;
    std::cout << "Program terminated." << std::endl;

    return 0;
}