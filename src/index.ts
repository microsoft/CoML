import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { ISessionContext } from '@jupyterlab/apputils';
import {
  INotebookTracker,
  Notebook,
  NotebookActions
} from '@jupyterlab/notebook';
import { OutputArea } from '@jupyterlab/outputarea';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { CodeCell } from '@jupyterlab/cells';

function findCellByOutputArea(
  notebook: Notebook,
  outputArea: OutputArea
): number {
  // We handle the situation when Shift+Enter is pressed and active cell is not the last one.
  const cells = notebook.widgets;

  const matches: number[] = [];
  for (let i = 0; i < cells.length; i++) {
    const cell = cells[i];
    if (
      cell.model.type === 'code' &&
      (cell as CodeCell).outputArea === outputArea
    ) {
      matches.push(i);
    }
  }

  if (matches.length === 1) {
    return matches[0];
  } else if (matches.length > 1) {
    // Multiple match, we take the one closest to (but not exceeding) to the active cell.
    let bestIndex = -1;
    for (const index of matches) {
      if (bestIndex === -1 || index <= notebook.activeCellIndex) {
        bestIndex = index;
      } else {
        break;
      }
    }
    return bestIndex;
  } else {
    // When something is wrong, we fallback to the active cell.
    return notebook.activeCell ? notebook.activeCellIndex : -1;
  }
}

function getNotebookContext(
  app: JupyterFrontEnd,
  notebookTracker: INotebookTracker | null
): ISessionContext | undefined {
  if (!notebookTracker?.currentWidget) {
    return undefined;
  }
  if (notebookTracker.currentWidget !== app.shell.currentWidget) {
    return undefined;
  }
  return notebookTracker.currentWidget.context.sessionContext;
}

function getNotebook(
  app: JupyterFrontEnd,
  notebookTracker: INotebookTracker | null
): Notebook | undefined {
  if (!notebookTracker?.currentWidget) {
    return undefined;
  }
  if (notebookTracker.currentWidget !== app.shell.currentWidget) {
    return undefined;
  }
  return notebookTracker.currentWidget.content;
}

function insertCellBelow(
  notebook: Notebook,
  activeCellIndex: number, // Active cell index from notebook is not necessarily reliable
  code: string,
  chatMetadata: any = {},
  editorFocus = true
): void {
  notebook.model?.sharedModel.insertCell(activeCellIndex + 1, {
    cell_type: 'code',
    metadata: {
      coml: chatMetadata
    },
    source: code
  });
  notebook.activeCellIndex = activeCellIndex + 1;
  if (editorFocus) {
    notebook.activeCell?.editor?.focus();
  }
}

function getLastCell(notebook: Notebook, currentCellIndex: number) {
  if (currentCellIndex <= 0) {
    console.warn(
      `Current cell index is ${currentCellIndex}. No last cell found.`
    );
    return null;
  } else {
    const lastCell = notebook.widgets[currentCellIndex - 1];
    return lastCell.model;
  }
}

function getCurrentCell(notebook: Notebook, currentCellIndex: number) {
  if (currentCellIndex < 0) {
    console.warn(`Invalid current cell index: ${currentCellIndex}.`);
    return null;
  } else {
    const lastCell = notebook.widgets[currentCellIndex];
    return lastCell.model;
  }
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'coml:plugin',
  description: 'JupyterLab extension for CoML.',
  autoStart: true,
  optional: [INotebookTracker],
  activate: (
    app: JupyterFrontEnd,
    notebookTracker: INotebookTracker | null
  ) => {
    function handleCommand(
      outputArea: OutputArea,
      command: any,
      sendCallback: (msg: string) => void
    ): any {
      if (command['command'] === 'insert_cell_below') {
        // Command format: { "command": "insert_cell_below", "code": "print('hello')", "metadata": { "request": ... } }
        const notebook = getNotebook(app, notebookTracker);
        if (notebook) {
          const insertIndex = findCellByOutputArea(notebook, outputArea);
          insertCellBelow(
            notebook,
            insertIndex,
            command['code'],
            command['metadata']
          );
        } else {
          console.warn('No notebook found');
        }
        // Reply with empty string to indicate that the command is handled.
        return '';
      } else if (command['command'] === 'insert_and_execute_cell_below') {
        // Command format: { "command": "insert_and_execute_cell_below", "code": "print('hello')", "metadata": { "request": ... } }
        const notebook = getNotebook(app, notebookTracker);
        const sessionContext = getNotebookContext(app, notebookTracker);
        if (notebook && sessionContext) {
          const insertIndex = findCellByOutputArea(notebook, outputArea);
          insertCellBelow(
            notebook,
            insertIndex,
            command['code'],
            command['metadata'],
            false
          );
          // Reply must be sent before running the next cell to avoid the deadlock warning.
          sendCallback('');
          // Run active cell
          NotebookActions.run(notebook, sessionContext);
        } else {
          console.warn('No notebook or session context found');
          // Reply with empty string to indicate that the command is handled.
          return '';
        }
      } else if (command['command'] === 'last_cell') {
        // Command format: { "command": "last_cell" }
        const notebook = getNotebook(app, notebookTracker);
        if (notebook) {
          const currentCellIndex = findCellByOutputArea(notebook, outputArea);
          const lastCell = getLastCell(notebook, currentCellIndex);
          if (lastCell) {
            return JSON.stringify(lastCell.toJSON());
          } else {
            console.warn('No last cell found');
          }
        } else {
          console.warn('No notebook found');
        }
      } else if (command['command'] === 'running_cell') {
        // Command format: { "command": "running_cell" }
        const notebook = getNotebook(app, notebookTracker);
        if (notebook) {
          const currentCellIndex = findCellByOutputArea(notebook, outputArea);
          const cell = getCurrentCell(notebook, currentCellIndex);
          if (cell) {
            return JSON.stringify(cell.toJSON());
          } else {
            console.warn('No running cell is found');
          }
        } else {
          console.warn('No notebook found');
        }
      } else if (command['command'] === 'update_running_cell_metadata') {
        // Command format: { "command": "update_running_cell_metadata", "metadata": ... }
        const notebook = getNotebook(app, notebookTracker);
        if (notebook) {
          const currentCellIndex = findCellByOutputArea(notebook, outputArea);
          const cell = getCurrentCell(notebook, currentCellIndex);
          if (cell) {
            cell.setMetadata('coml', command['metadata']);
            return '';
          } else {
            console.warn('No running cell is found');
          }
        } else {
          console.warn('No notebook found');
        }
      } else {
        console.warn('Invalid command:', command);
        return undefined;
      }
      return '';
    }

    function hackedOnInputRequest(
      this: OutputArea,
      msg: KernelMessage.IInputRequestMsg,
      future: Kernel.IShellFuture
    ): void {
      // This is the hacked version of handler of `input()` (at kernel side).
      // Everything needs to be done at JS side is firstly sent here and routed within this method.

      let sent = false;
      function sendCallback(reply: string) {
        if (sent) {
          console.warn('Reply already sent.');
        } else {
          future.sendInputReply({ status: 'ok', value: reply }, msg.header);
          sent = true;
        }
      }

      try {
        // only apply the hack if the command is valid JSON
        const command = JSON.parse(msg.content.prompt);
        const result = handleCommand(this, command, sendCallback);
        if (result !== undefined) {
          if (!sent) {
            sendCallback(result);
          }
        } else {
          return (this as any).nativeOnInputRequest(msg, future);
        }
      } catch (err) {
        console.log('Not a JSON command', msg, err);
        return (this as any).nativeOnInputRequest(msg, future);
      }
    }

    (OutputArea.prototype as any).nativeOnInputRequest = (
      OutputArea.prototype as any
    ).onInputRequest;
    (OutputArea.prototype as any).onInputRequest = hackedOnInputRequest;

    app.commands.addCommand('coml:insert_cell_below', {
      label: 'Execute coml:insert_cell_below Command',
      caption: 'Execute coml:insert_cell_below Command',
      execute: (args: any) => {
        const notebook = getNotebook(app, notebookTracker);
        if (!notebook) {
          console.warn('No notebook found');
          return;
        }
        insertCellBelow(
          notebook,
          notebook.activeCellIndex,
          args['code'],
          args['metadata']
        );
      }
    });

    app.commands.addCommand('coml:insert_and_execute_cell_below', {
      label: 'Execute coml:insert_and_execute_cell_below Command',
      caption: 'Execute coml:insert_and_execute_cell_below Command',
      execute: (args: any) => {
        const notebook = getNotebook(app, notebookTracker);
        const sessionContext = getNotebookContext(app, notebookTracker);
        if (!notebook || !sessionContext) {
          console.warn('Notebook or session context not found');
          return;
        }
        insertCellBelow(
          notebook,
          notebook.activeCellIndex,
          args['code'],
          args['metadata']
        );
        NotebookActions.run(notebook, sessionContext);
      }
    });
  }
};

export default plugin;
