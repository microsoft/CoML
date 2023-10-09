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
      chatdata: chatMetadata
    },
    source: code
  });
  notebook.activeCellIndex = activeCellIndex + 1;
  if (editorFocus) {
    notebook.activeCell?.editor?.focus();
  }
}

function getLastCell(notebook: Notebook, currentCellIndex: number): any {
  if (currentCellIndex <= 0) {
    console.warn(
      `Current cell index is ${currentCellIndex}. No last cell found.`
    );
    return null;
  } else {
    const lastCell = notebook.widgets[currentCellIndex - 1];
    return lastCell.model.toJSON();
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
    function hackedOnInputRequest(
      this: OutputArea,
      msg: KernelMessage.IInputRequestMsg,
      future: Kernel.IShellFuture
    ): void {
      // This is the hacked version of handler of `input()` (at kernel side).
      // Everything needs to be done at JS side is firstly sent here and routed within this method.
      try {
        // only apply the hack if the command is valid JSON
        const command = JSON.parse(msg.content.prompt);
        if (command['command'] === 'insert_cell_below') {
          // Command format: { "command": "insert_cell_below", "code": "print('hello')", "metadata": { "request": ... } }
          const notebook = getNotebook(app, notebookTracker);
          if (notebook) {
            const insertIndex = findCellByOutputArea(notebook, this);
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
          future.sendInputReply({ status: 'ok', value: '' }, msg.header);
        } else if (command['command'] === 'insert_and_execute_cell_below') {
          // Command format: { "command": "insert_and_execute_cell_below", "code": "print('hello')", "metadata": { "request": ... } }
          const notebook = getNotebook(app, notebookTracker);
          const sessionContext = getNotebookContext(app, notebookTracker);
          if (notebook && sessionContext) {
            const insertIndex = findCellByOutputArea(notebook, this);
            insertCellBelow(
              notebook,
              insertIndex,
              command['code'],
              command['metadata'],
              false
            );
            // Reply must be sent before running the next cell to avoid the deadlock warning.
            future.sendInputReply({ status: 'ok', value: '' }, msg.header);
            // Run active cell
            NotebookActions.run(notebook, sessionContext);
          } else {
            console.warn('No notebook or session context found');
            // Reply with empty string to indicate that the command is handled.
            future.sendInputReply({ status: 'ok', value: '' }, msg.header);
          }
        } else if (command['command'] === 'last_cell') {
          // Command format: { "command": "last_cell" }
          const notebook = getNotebook(app, notebookTracker);
          if (notebook) {
            const currentCellIndex = findCellByOutputArea(notebook, this);
            const lastCell = getLastCell(notebook, currentCellIndex);
            future.sendInputReply(
              { status: 'ok', value: JSON.stringify(lastCell) },
              msg.header
            );
          } else {
            console.warn('No notebook found');
          }
          future.sendInputReply({ status: 'ok', value: 'hello' }, msg.header);
        } else {
          console.warn('Not a command', msg);
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
