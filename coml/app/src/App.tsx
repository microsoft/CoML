import { vscode } from "./utilities/vscode";
import { VSCodeButton, VSCodeTextField, VSCodeTextArea, VSCodeDivider, VSCodeLink } from "@vscode/webview-ui-toolkit/react";
import "./App.css";
import "@vscode/codicons/dist/codicon.css";
import { useEffect, useRef, useState } from "react";
import autosize from "autosize";
import ReactMarkdown from 'react-markdown'

import { chatWithGPT, suggestMachineLearningModule, prepareCache } from "./chatml";
import { HumanMessage, SystemMessage, BaseMessage, FunctionMessage, AIMessage } from "langchain/schema";

async function chatNow(messages: BaseMessage[], setMessages: (messages: BaseMessage[]) => void) {
  const logTheme = "color: #854085";
  try {
    const response = await chatWithGPT(messages);
    if (response.additional_kwargs && response.additional_kwargs.function_call) {
      const functionCall = response.additional_kwargs.function_call;
      if (functionCall.name === "suggestMachineLearningModule" && functionCall.arguments) {
        console.log(`${logTheme}Function call: suggestMachineLearningModule. Arguments:\n${functionCall.arguments}`);
        const args = JSON.parse(functionCall.arguments);
        if ("existingModules" in args && "targetRole" in args) {
          const schema = args.targetSchemaId ? args.targetSchemaId : undefined;
          const result = await suggestMachineLearningModule(args.existingModules, args.targetRole, schema);
          if (result) {
            setMessages([...messages, new FunctionMessage({
              name: functionCall.name,
              content: JSON.stringify(result)
            }, "")]);
          }
        }
        return;
      }
    } else {
      console.log(`${logTheme}No function call`);
      setMessages([...messages, response.content ? response : new AIMessage("No response.")]);
    }
    console.log(response);
    setMessages([...messages, response]);
  } catch (e) {
    console.error(e);
    setMessages([...messages, new SystemMessage("Error: " + e)]);
  }
}

function App()  {
  const [messages, setMessages] = useState<BaseMessage[]>([]);
  const [generating, setGenerating] = useState<boolean>(false);

  function addHumanMessage(textInput: string) {
    console.log("Received input: " + textInput);
    (textareaRef.current as any).value = "";  // clear input
    autosize((textareaRef.current as any).shadowRoot.querySelectorAll("textarea"));
    setMessages([...messages, new HumanMessage(textInput)]);
  }

  function captureKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.code === "Enter" && !event.shiftKey) {
      event.preventDefault();
      addHumanMessage((textareaRef.current as any).value);
    }
  }

  useEffect(() => {
    autosize((textareaRef.current as any).shadowRoot.querySelectorAll("textarea"));
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage instanceof HumanMessage || lastMessage instanceof FunctionMessage) {
        setGenerating(true);
        chatNow(messages, setMessages).then(() => {
          setGenerating(false);
        });
      }
    }
  }, [messages]);

  useEffect(() => {
    // prepareCache();
    // suggestMachineLearningModule([
    //   {
    //     role: "dataset",
    //     module: {
    //       name: "MNIST",
    //       description: "A dataset of handwritten digits",
    //     }
    //   },
    // ], "verifiedAlgorithm", "rpart-preproc-4796");
    // suggestMachineLearningModule([
    //   {
    //     role: "model",
    //     module: {
    //       name: "ViT-Base",
    //       description: "A transformer model for image classification",
    //     }
    //   },
    //   {
    //     role: "dataset",
    //     module: {
    //       name: "ImageNet",
    //       description: "A dataset of images",
    //     }
    //   }
    // ], "algorithm")
    // suggestMachineLearningModule([
    //   {"role": "taskType", "module": "image-classification"},
    //   {"role": "dataset", "module": "plantvillage"}
    // ], "algorithm")
    // queryEmbedding("Hello world 2!");
  }, [])

  const textareaRef = useRef(null);

  const messageDisplay: JSX.Element[] = [];
  for (let i = 0; i < messages.length; ++i) {
    if (messages[i] instanceof HumanMessage) {
      messageDisplay.push(<h3 key={`message-${i}-role`}>User</h3>);
    } else if (messages[i] instanceof FunctionMessage) {
      messageDisplay.push(<h3 key={`message-${i}-role`}>ML Expert</h3>);
    } else if (messages[i] instanceof AIMessage) {
      messageDisplay.push(<h3 key={`message-${i}-role`}>Assistant</h3>);
    }
    messageDisplay.push(<ReactMarkdown key={`message-${i}-content`}>{messages[i].content}</ReactMarkdown>);
    if (i < messages.length - 1) {
      messageDisplay.push(<VSCodeDivider key={`message-${i}-divider`} />);
    }
  }

  const inspiringText = "Recommend a config of rpart.preproc algorithm for MNIST dataset, a dataset of handwritten digits.";

  return (
    <main>
      <h1>ChatML</h1>
      <div className="container">
        {messageDisplay}
        {!generating && <div>
          <VSCodeLink href="#" onClick={() => addHumanMessage(inspiringText)}>
            <span className="lightbulb codicon codicon-lightbulb"></span>
            {inspiringText}
          </VSCodeLink>
          <br/>
          {messages.length > 0 && <VSCodeLink href="#" onClick={() => setMessages([])}>
            <span className="clear-all codicon codicon-clear-all"></span>
            Restart conversation
          </VSCodeLink>}
        </div>}
      </div>
      <div className="chat-box">
        <span
          className="send codicon codicon-send"
          onClick={() => addHumanMessage((textareaRef.current as any).value)}
        ></span>
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
