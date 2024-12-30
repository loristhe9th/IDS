# IDS
# Network Attacks Analysis and Suggestion for Simple Intrusion Detection System

## 1. Overview
### 1.1 Title
Network Attacks Analysis and Suggestion for Simple Intrusion Detection System

### 1.2 Objective
- Analyze network attack data to understand patterns and risks.  
- Propose a feature for a simple and effective network intrusion detection system (IDS).  

### 1.3 Tools and Technologies Used
- **Programming Language:** Python  
- **Framework:** Streamlit (for building a web-based interface)  
- **Data Source:** CICIDS2017 dataset  
- **Diagram Tools:** Draw.io (to create the Use Case Diagram)  

---

## 2. Use Case Diagram (see UseCase.png)
### 2.1 Actors
1. **System Administrator**:  
   - Manages and configures the intrusion detection system (IDS).  
   - Reviews logs and monitors the system's performance.  
2. **Network User**:  
   - Uses the network being monitored by the IDS.  
   - Receives alerts for potential intrusions.  
3. **Intrusion Detection System (IDS)**:  
   - Automatically detects anomalies in network traffic.  
   - Logs incidents and generates alerts for detected threats.  

### 2.2 Use Cases
1. **Detect Intrusion**:  
   - The IDS automatically analyzes network traffic to identify potential threats.  
2. **Generate Alert**:  
   - The IDS notifies relevant parties when an intrusion is detected.  
   - Logs the incident as part of the process.  
3. **Log Incident**:  
   - The IDS records detected threats and corresponding actions for future analysis.  
4. **Review Logs**:  
   - The System Administrator evaluates logs to improve security measures.  
5. **Receive Alert**:  
   - Alerts are sent to Network Users or System Administrators regarding potential intrusions.  

### 2.3 Relationships
- `Detect Intrusion` → `Generate Alert`: Detection is a prerequisite for alert generation.  
- `Generate Alert` → `Log Incident`: Every alert is logged for documentation.  
- `Log Incident` → `Review Logs`: Logs serve as the foundation for review and analysis.  
- **Actor Relationships**:  
   - IDS interacts directly with both the System Administrator and Network User by providing alerts and logs.  
  

---

## 3. Data and Methodology
### 3.1 Data Source and Trained Model
#### **Data Source**
- **Dataset**: CICIDS2017  
- **Format**: CSV file containing network traffic logs.  

#### **Dataset Overview**
- **Number of Samples**: Over 3 million rows.  
- **Number of Features**: ~80 columns (e.g., Flow ID, Protocol, Packet Statistics, Flow Bytes/s).  
- **Attack Types**: 15 types, including DoS, DDoS, Brute Force Attack, SQL Injection, and more.  

#### **Trained Model**
- **Algorithm Used**: K-Nearest Neighbors (KNN).  
- Adapted from the SafeML project and fine-tuned for this IDS.  

### 3.2 Suggested Feature of Intrusion Detection System
- **Script**: `app.py` (available in this repository).  
- **Functionality**:  
  - Processes incoming network data in real-time.  
  - Identifies anomalies using the pre-trained model.  
  - Generates alerts based on predefined thresholds.  
- **User Interface**: Built with Streamlit, providing features such as:  
  - Real-time monitoring.  
  - Log visualization.  

---

## 4. Implementation
### 4.1 Architecture Diagram
- Data Flow: From the data source → Pre-trained Model → Early Warning System → User Interface.  

### 4.2 Key Features
1. Real-time monitoring of network traffic.  
2. Customizable thresholds for intrusion alerts.  
3. Logging of detected threats for review and analysis.  

---

## 5. Repository Structure
Repository Structure:
├── Data
│   ├── Friday-WorkingHours-Afternoon.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Intrusion Detection System
│   └── app.py  # Main Streamlit app for intrusion detection
├── Network Attacks Analysis
│   └── CICIDS2017_Data Analysis.ipynb
│   └── Requirements  # List of dependencies for the project
│   └── knn_model.pkl # Trained KNN model for intrusion detection
├── UseCase.png # Use Case Diagram created using Draw.io
├── LICENSE
├── README.md
└── .gitattributes
