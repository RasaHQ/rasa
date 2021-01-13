// @flow
import React, { Component } from "react";
import { FaMicrophone, FaCircle } from "react-icons/fa";

type SpeechInputProps = {
  onSpeechInput: (message: string) => Promise<void>,
  onSpeechEnd: () => Promise<void>,
  language: string
};

type SpeechInputState = {
  isRecognizing: boolean,
  finalTranscript: string,
  interimTranscript: string
};

function supportsSpeechRecognition() {
  return "webkitSpeechRecognition" in window;
}

const first_char = /\S/;
function capitalize(s) {
  return s.replace(first_char, m => m.toUpperCase());
}

export default class SpeechInput extends Component<
  SpeechInputProps,
  SpeechInputState
> {
  static defaultProps = {
    language: "en-US"
  };
  state = {
    isRecognizing: false,
    finalTranscript: "",
    interimTranscript: ""
  };
  recognition: any;
  componentDidMount() {
    if (!supportsSpeechRecognition()) return;

    // $FlowFixMe We're checking for the existence of the API before
    const recognition = new webkitSpeechRecognition();
    recognition.lang = this.props.language;
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.onstart = this.onRecognitionStart;
    recognition.onerror = this.onRecognitionError;
    recognition.onend = this.onRecognitionEnd;
    recognition.onresult = this.onRecognitionResult;
    this.recognition = recognition;
  }

  onRecognitionStart = () => {
    this.setState({
      isRecognizing: true,
      finalTranscript: "",
      interimTranscript: ""
    });
  };

  onRecognitionResult = (event: any) => {
    this.setState(prevState => {
      let { finalTranscript } = prevState;
      let interimTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }
      this.props.onSpeechInput(capitalize(finalTranscript + interimTranscript));
      return { finalTranscript, interimTranscript };
    });
  };

  onRecognitionEnd = () => {
    this.setState({ isRecognizing: false });
    this.recognition.stop();
    this.props.onSpeechEnd();
  };

  onRecognitionError = (event: { error: Error }) => {
    // TODO: Display proper error, but where?
    console.error(event.error);
  };

  startRecognition = () => {
    if (this.state.isRecognizing) {
      this.recognition.stop();
      return;
    }

    this.recognition.start();
  };

  render() {
    return supportsSpeechRecognition() ? (
      <button type="button" id="speech-input" onClick={this.startRecognition}>
        {this.state.isRecognizing ? (
          <FaCircle className="vertical-center" color="#ed4933" />
        ) : (
          <FaMicrophone className="vertical-center" />
        )}
      </button>
    ) : null;
  }
}
