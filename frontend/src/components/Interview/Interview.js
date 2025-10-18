/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React, { useState, useEffect, useRef } from "react";
import { marked } from "marked";
import parse from "html-react-parser";
import { diffArrays, diffWords } from "diff";
import "./Interview.css";
import DetailsPopup from "../DetailsPopup/DetailsPopup";

const Interview = ({ selectedPatient, selectedCondition, onBack }) => {
  const [messages, setMessages] = useState([]);
  const [isInterviewComplete, setIsInterviewComplete] = useState(false);
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [evaluation, setEvaluation] = useState('');
  const [isFetchingEvaluation, setIsFetchingEvaluation] = useState(false);
  const [currentReport, setCurrentReport] = useState("");
  const [prevReport, setPrevReport] = useState("");
  const [waitTime, setWaitTime] = useState(3000);
  const [showEvaluationInfoPopup, setShowEvaluationInfoPopup] = useState(false);
  const [isDetailsPopupOpen, setIsDetailsPopupOpen] = useState(false);
  const chatContainerRef = useRef(null);
  const reportContentRef = useRef(null);
  const lastMessageRef = useRef(null);
  const messageQueue = useRef([]);
  const eventSourceRef = useRef(null);
  const timeoutIdRef = useRef(null);

  const currentPlayingAudio = useRef(null); // To keep track of the currently playing audio instance
  const isAudioEnabledRef = useRef(isAudioEnabled);
  useEffect(() => {
    isAudioEnabledRef.current = isAudioEnabled;
  }, [isAudioEnabled]);
  const waitTimeRef = useRef(waitTime);
  useEffect(() => {
    waitTimeRef.current = waitTime;
  }, [waitTime]);

  const processQueue = React.useCallback(() => {
    if (timeoutIdRef.current) {
      clearTimeout(timeoutIdRef.current);
    }

    if (messageQueue.current.length === 0) {
      // The queue is empty, so the processing chain for this batch is done.
      // Clear the timeout ref so a new message can start a new chain.
      timeoutIdRef.current = null;
      setIsInterviewComplete(
        eventSourceRef.current && eventSourceRef.current.readyState === EventSource.CLOSED
      );
      return;
    }

    const nextMessage = messageQueue.current.shift();

    setMessages((prev) => [...prev, nextMessage]);

    if (nextMessage.audio && isAudioEnabledRef.current) {
      if (currentPlayingAudio.current) {
        currentPlayingAudio.current.pause();
        currentPlayingAudio.current.src = '';
      }
      const audio = new Audio(nextMessage.audio);
      currentPlayingAudio.current = audio;

      audio.onended = () => {
        currentPlayingAudio.current = null;
        processQueue();
      };
      audio.onerror = (e) => {
        console.error("Audio playback error:", e);
        currentPlayingAudio.current = null;
        processQueue();
      };
      audio.play().catch(e => {
        console.error("Error playing audio automatically:", e);
        currentPlayingAudio.current = null;
        processQueue();
      });
    } else {
      // For non-audio, schedule the next processing call with a fixed delay
      // to simulate reading time. This will call processQueue again, which will
      // handle an empty queue and stop the chain if needed.
      timeoutIdRef.current = setTimeout(processQueue, waitTimeRef.current);
    }
  }, [setMessages, setIsInterviewComplete]);

  useEffect(() => {
    if (!selectedPatient || !selectedCondition) return;

    setMessages([]);
    setIsInterviewComplete(false);
    messageQueue.current = [];
    if (currentPlayingAudio.current) {
      currentPlayingAudio.current.pause();
      currentPlayingAudio.current = null;
    }
    // Prepend base URL if running on localhost:3000
    const baseURL =
      window.location.origin === "http://localhost:3000"
        ? "http://localhost:7860"
        : "";
    const url = `${baseURL}/api/stream_conversation?patient=${encodeURIComponent(
      selectedPatient.name
    )}&condition=${encodeURIComponent(selectedCondition)}`;
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Check if the parsed object is our special 'end' signal
        if (data && data.event === 'end') {
          console.log("Server signaled end of stream. Closing connection.");
          eventSource.close();
          processQueue();
          return; 
        }        
        messageQueue.current.push(data);
        // Always call processQueue after pushing a message, unless audio or timeout is active
        if (!currentPlayingAudio.current && !timeoutIdRef.current) {
          processQueue();
        }
      } catch (error) {
        console.warn("Could not parse message data. Data received:", event.data, "Error:", error);
      }
    };

    eventSource.onerror = (err) => {
      console.error("EventSource failed:", err);
      eventSource.close();
    };


    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (timeoutIdRef.current) {
        clearTimeout(timeoutIdRef.current);
        timeoutIdRef.current = null;
      }
      // Ensure any playing audio is stopped when component unmounts or dependencies change
      if (currentPlayingAudio.current) {
        currentPlayingAudio.current.pause();
        currentPlayingAudio.current = null;
      }
    };
  }, [selectedPatient, selectedCondition, processQueue]);

  useEffect(() => {
    processQueue();
  }, [waitTime, processQueue]);
  
  useEffect(() => {
    // Prevent body scroll when Interview is shown
    document.body.style.overflowY = "clip";
    return () => {
      document.body.style.overflowY = "unset";
    };
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      const container = chatContainerRef.current;
      const lastMessage = messages[messages.length - 1];
      if (lastMessage && lastMessage.speaker === "report") { 
        return;
      }

      const isNearBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight <
        container.clientHeight;
      if (isNearBottom && messages.length > 0) {
        lastMessageRef.current.scrollIntoView({
          behavior: "smooth",
          block: "end",
        });
      }
    }
  }, [messages]);

  // Update report on new messages
  useEffect(() => {
    const reportMessages = messages.filter((msg) => msg.speaker === "report");
    if (reportMessages.length > 0) {
      const latestReportMessageText =
        reportMessages[reportMessages.length - 1].text;
      const newReport = marked(latestReportMessageText.trim());
      if (newReport !== currentReport) {
        setPrevReport(currentReport);
        setCurrentReport(newReport);
      }
    }
  }, [messages, currentReport]);

  // Updated diff function to tokenize HTML and use nested diffWords for text changes
  const getDiffReport = () => {
    // Tokenize HTML into tags and text parts
    const tokenizeHTML = (html) => html.match(/(<[^>]+>|[^<]+)/g) || [];
    const tokensPrev = tokenizeHTML(prevReport);
    const tokensCurrent = tokenizeHTML(currentReport);
    const diffParts = diffArrays(tokensPrev, tokensCurrent);

    let result = "";
    for (let i = 0; i < diffParts.length; i++) {
      // If a removed part is immediately followed by an added part,
      // and both are plain text (not an HTML tag), apply inner diffWords.
      if (
        diffParts[i].removed &&
        i + 1 < diffParts.length &&
        diffParts[i + 1].added
      ) {
        const removedText = diffParts[i].value.join("");
        const addedText = diffParts[i + 1].value.join("");
        // Check if both parts are not HTML tags
        if (
          (!/^<[^>]+>$/.test(removedText) && !/^<[^>]+>$/.test(addedText))
        ) {
          const innerDiff = diffWords(removedText, addedText);
          const innerResult = innerDiff
            .map((part) => {
              if (part.added) {
                return `<span class="add">${part.value}</span>`;
              } else if (part.removed) {
                return `<span class="remove">${part.value}</span>`;
              }
              return part.value;
            })
            .join("");
          result += innerResult;
          i++;
          continue;
        }
      }
      if (diffParts[i].added) {
        result += `<span class="add">${diffParts[i].value.join("")}</span>`;
      } else if (diffParts[i].removed) {
        result += `<span class="remove">${diffParts[i].value.join("")}</span>`;
      } else {
        result += diffParts[i].value.join("");
      }
    }
    return result;
  };

  // Fetch evaluation when showEvaluation is triggered
  useEffect(() => {
    if (!showEvaluation) return;
    setIsFetchingEvaluation(true);
    setEvaluation('');
    // Get latest report
    const reportMessages = messages.filter((msg) => msg.speaker === "report");
    const report =
      reportMessages.length > 0
        ? marked(reportMessages[reportMessages.length - 1].text.trim())
        : "<p>No report available.</p>";
    // Prepend base URL if running on localhost:3000
    const baseURL = window.location.origin === "http://localhost:3000" ? "http://localhost:7860" : "";
    fetch(`${baseURL}/api/evaluate_report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        report,
        condition: selectedCondition
      })
    })
      .then(response => response.json())
      .then(data => {
        setEvaluation(data.evaluation.replace('```html\n','').replace('\n```',''));
        setIsFetchingEvaluation(false);
      })
      .catch(error => {
        setEvaluation('Error fetching evaluation.');
        setIsFetchingEvaluation(false);
      });
  }, [showEvaluation, messages, selectedCondition]);

  // Scroll report-content to bottom when evaluate button appears
  useEffect(() => {
    if (isInterviewComplete && reportContentRef.current) {
      reportContentRef.current.scrollTop = reportContentRef.current.scrollHeight;
    }
  }, [isInterviewComplete]);

  const handleToggleWaitTime = () => {
    setWaitTime((prev) => (prev === 1000 ? 3000 : 1000));
  };

  const handleToggleAudio = () => {
    setIsAudioEnabled(prev => {
      const isNowEnabled = !prev;
      // If we are disabling audio and something is playing, stop it and continue the queue.
      if (!isNowEnabled && currentPlayingAudio.current) {
        currentPlayingAudio.current.pause();
        currentPlayingAudio.current.src = '';
        currentPlayingAudio.current = null;
      }
      return isNowEnabled;
    });
  };

  const playAudio = (audioDataUrl) => {
    if (audioDataUrl) {
      const audio = new Audio(audioDataUrl);
      audio.play().catch(e => {
        console.error("Error playing audio:", e);
      });
    }
  };

  return (
    <div className="page interview-page">
      <div className="headerButtonsContainer">
        <button className="back-button" onClick={onBack}>
          <i className="material-icons back-button-icon">keyboard_arrow_left</i>
          Back
        </button>
        <button className="details-button" onClick={() => setIsDetailsPopupOpen(true)}>
          <i className="material-icons code-block-icon">code</i>&nbsp; Details
          about this Demo
        </button>
      </div>
      <div className="frame">
        <div className="interview-split-container">
          {/* Top: Interview Chat */}
          <div className="interview-left-section">
            {/* Right: Chat */}
            <div className="interview-chat-panel">
              <div className="header2">
                Simulated Interview
                &nbsp;
                <i
                  className="material-icons toggle-icon"
                  style={{
                    cursor: "pointer",
                    color: isAudioEnabled ? "#1976d2" : "#888",
                  }}
                  title={`Click to ${
                    isAudioEnabled ? "disable" : "enable"
                  } audio`}
                  onClick={handleToggleAudio}
                >
                  {isAudioEnabled ? "volume_up" : "volume_off"}
                </i>
                {isAudioEnabled && (<span>audio by Gemini TTS</span>)}
                {!isAudioEnabled && (
                <i
                  className="material-icons toggle-icon"
                  style={{
                    cursor: "pointer",
                    color: waitTime === 1000 ? "#1976d2" : "#888",
                  }}
                  title={`Click to ${
                    waitTime === 1000 ? "slow down" : "speed up"
                  } the interview`}
                  onClick={handleToggleWaitTime}
                >
                  speed
                </i>)}
                
              </div>
              <div className="chat-container" ref={chatContainerRef}>
                {messages.length === 0 ? (
                  <div className="chat-waiting-indicator">
                    Waiting for the interview to start...
                  </div>
                ) : (
                  messages
                    .filter((msg) => msg.speaker !== "report")
                    .map((msg, idx, filteredMessages) => (
                      <div
                        ref={idx === filteredMessages.length - 1 ? lastMessageRef : null}
                        className={`chat-message-wrapper ${msg.speaker}${idx === filteredMessages.length - 1 ? " fade-in" : ""}${msg.audio ? " has-audio" : ""}`}
                      key={idx}
                    >
                      {msg.speaker.includes("interviewer") && (
                        <img
                          className="chat-avatar"
                          src="assets/ai_headshot.svg"
                          alt="Interviewer"
                        />
                      )}
                      <div className={`chat-bubble ${msg.audio ? "with-audio" : ""}`}>
                        {msg.speaker.includes("thinking") && (
                          <div className="thinking-header">Thinking...</div>
                        )}
                        {msg.text}
                      </div>
                      {msg.speaker === "patient" && (
                        <img
                          className="chat-avatar"
                          src={selectedPatient.headshot}
                          alt={selectedPatient.name}
                        />
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
          {/* Right: Report Section */}
          <div className="interview-right-section">
            <div className="header2">Generated Report</div>
            <div className="report-content" ref={reportContentRef}>
              {/* Updated report rendering to show diff if available */}
              <div
                dangerouslySetInnerHTML={{
                  __html: prevReport ? getDiffReport() : currentReport,
                }}
              />
              {isInterviewComplete && (
                <button
                  className="evaluate-button"
                  onClick={() => setShowEvaluationInfoPopup(true)}
                  disabled={showEvaluation || showEvaluationInfoPopup}
                ><i className="material-icons back-button-icon">keyboard_arrow_down</i>
                  View Report Evaluation
                </button>
              )}
              <div className="evaluation-text">
                {showEvaluation && (
                  isFetchingEvaluation
                    ? <div>Please wait...</div>
                    : parse(evaluation)
                )}
              </div>
            </div>
            <div className="disclaimer-container">
              <i className="material-icons warning-icon">warning</i>
              <div className="disclaimer-text">
                This demonstration is for illustrative purposes of MedGemma’s baseline capabilities only. It does not represent a finished or approved product, is not intended to diagnose or suggest treatment of any disease or condition, and should not be used for medical advice.
              </div>
            </div>
          </div>
        </div>
      </div>
      {showEvaluationInfoPopup && (
        <div className="popup-overlay">
          <div className="popup-content">
            <h2>About the Evaluation</h2>
            <p>
              Now we will ask MedGemma to evaluate its own performance at
              generating this report. We will provide it with all the
              information about {selectedPatient.name}, including their actual
              diagnosis and aspects of condition history not included previously. 
              Using this new information, MedGemma will
              highlight key facts it correctly included and identify other
              information that would have been beneficial to add.
            </p>
            <p>
              The purpose of this step is to provide non-medical users with a
              sense of how well MedGemma did at this task. While the evaluation
              is completed by MedGemma, the examples in this demo have also been
              reviewed by clinicians for accuracy. Although MedGemma's evaluation
              does not represent a consensus based standard,
              this illustration simply shows an example of one approach developers could adopt 
              to evaluate quality and completeness.
            </p>
            <button className="popup-button" onClick={() => {
              setShowEvaluationInfoPopup(false);
              setShowEvaluation(true);
            }}>Continue</button>
          </div>
        </div>)}
      <DetailsPopup
        isOpen={isDetailsPopupOpen}
        onClose={() => setIsDetailsPopupOpen(false)}
      />
    </div>
  );
};

export default Interview;
