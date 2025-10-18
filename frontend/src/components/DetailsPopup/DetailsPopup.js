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

import React from 'react';
import './DetailsPopup.css';

const DetailsPopup = ({ isOpen, onClose }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="popup-overlay" onClick={onClose}>
      <div className="popup-content" onClick={(e) => e.stopPropagation()}>
        <button className="popup-close-button" onClick={onClose}>&times;</button>
        <h2 id="dialog-title" className="dialog-title-text">Details About This Demo</h2>
                <p><b>The Model:</b> This demo features Google's MedGemma-27B, a Gemma 3-based model
                    fine-tuned for comprehending medical text. It demonstrates MedGemma's ability to
                    accelerate the development of AI-powered healthcare applications by offering advanced
                    interpretation of medical data.</p>
                <p><b>Accessing and Using the Model:</b> Google's MedGemma-27B is available on <a
                        href="https://huggingface.co/google/medgemma-27b-text-it" target="_blank" rel="noopener noreferrer">HuggingFace<img
                            className="hf-logo"
                            src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" />
                    </a> and is easily deployable via&nbsp;
                    <a href="https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma" target="_blank" rel="noopener noreferrer">Model
                        Garden <img className="hf-logo"
                            src="https://www.gstatic.com/cloud/images/icons/apple-icon.png" /></a>.
                    Learn more about using the model and its limitations on the <a
                        href="https://developers.google.com/health-ai-developer-foundations?referral=appoint-ready"
                        target="_blank" rel="noopener noreferrer">HAI-DEF
                        developer site</a>.
                </p>
                <p><b>Health AI Developer Foundations (HAI-DEF)</b> provides a collection of open-weight models and
                    companion resources to empower developers in building AI models for healthcare.</p>
                <p><b>Share this Demo:</b> If you find this demonstration valuable, we encourage you to share it on
                    social media.
                    <small>
                    &nbsp;<a href="https://www.linkedin.com/shareArticle?mini=true&url=https://huggingface.co/spaces/google/appoint-ready&text=%23MedGemma%20%23MedGemmaDemo" target="_blank" rel="noopener noreferrer">LinkedIn</a>
                    &nbsp;<a href="http://www.twitter.com/share?url=https://huggingface.co/spaces/google/appoint-ready&hashtags=MedGemma,MedGemmaDemo" target="_blank" rel="noopener noreferrer">X/Tweet</a>
                    </small>
                </p>
                <p><b>Explore More Demos:</b> Discover additional demonstrations on HuggingFace Spaces or via Colabs:
                </p>
                <ul>
                    <li><a href="https://huggingface.co/collections/google/hai-def-concept-apps-6837acfccce400abe6ec26c1"
                            target="_blank" rel="noopener noreferrer">
                            Collection of concept apps <img className="hf-logo" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" />
                        </a> built around HAI-DEF open models to inspire the community.</li>
                    <li><a href="https://github.com/Google-Health/medgemma/tree/main/notebooks/fine_tune_with_hugging_face.ipynb" target="_blank" rel="noopener noreferrer">
                            Finetune MedGemma Colab <img className="hf-logo"
                                src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" /></a>
                        -
                        See an example of how to fine-tune this model.</li> 
                </ul>
                For more technical details about this demo, please refer to the <a href="https://huggingface.co/spaces/google/appoint-ready/blob/main/README.md#table-of-contents" target="_blank" rel="noopener noreferrer">README</a> file in the repository.
        <button className="popup-button" onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export default DetailsPopup;
