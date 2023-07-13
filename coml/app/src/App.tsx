import { vscode } from "./utilities/vscode";
import { VSCodeButton, VSCodeTextField, VSCodeTextArea, VSCodeDivider } from "@vscode/webview-ui-toolkit/react";
import "./App.css";
import "@vscode/codicons/dist/codicon.css";
import { useEffect, useRef, useState } from "react";
import autosize from "autosize";

import { chatWithGPT } from "./chatml";
import { HumanMessage, SystemMessage, BaseMessage } from "langchain/schema";

function App(a)  {
  const [messages, setMessages] = useState<BaseMessage[]>([]);

  function submit() {
    const textInput = (textareaRef.current as any).value;
    console.log(textInput);
    (textareaRef.current as any).value = "";  // clear input
    autosize((textareaRef.current as any).shadowRoot.querySelectorAll("textarea"));
    setMessages([...messages, new HumanMessage(textInput)]);
  }

  function captureKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.code === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  useEffect(() => {
    autosize((textareaRef.current as any).shadowRoot.querySelectorAll("textarea"));
  });

  useEffect(() => {
    if (messages.length > 0 && (messages[messages.length - 1] instanceof HumanMessage)) {
      chatWithGPT(messages).then((response) => {
        setMessages([...messages, response]);
      });
    }
  }, [messages]);

  const textareaRef = useRef(null);

  const messageDisplay: JSX.Element[] = [];
  for (let i = 0; i < messages.length; ++i) {
    if (messages[i] instanceof HumanMessage) {
      messageDisplay.push(<h3 key={`message-${i}-role`}>User</h3>);
    } else {
      messageDisplay.push(<h3 key={`message-${i}-role`}>ChatML</h3>);
    }
    messageDisplay.push(<p key={`message-${i}-content`}>{messages[i].content}</p>);
    if (i < messages.length - 1) {
      messageDisplay.push(<VSCodeDivider key={`message-${i}-divider`} />);
    }
  }
  return (
    <main>
      <h1>ChatML</h1>
      {messageDisplay}
      <div className="chat-box">
        <span className="send codicon codicon-send" onClick={submit}></span>
        <VSCodeTextArea
          resize="none"
          className="text"
          name="text"
          placeholder="Ask anything"
          ref={textareaRef}
          rows={1}
          onKeyDown={captureKeyDown}
        />
      </div>
    </main>
  );
}

export default App;
