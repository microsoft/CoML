import { TableClient } from "@azure/data-tables";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { blobSasToken, blobTableName, blobUrl, openAIApiKey } from "./apiKey";

interface TableEntity {
    partitionKey: string,
    rowKey: string,
    embedding: Uint8Array,
}

let _prefetched: Map<string, Float32Array> | undefined = undefined;

function getTableClient() {
  return new TableClient(blobUrl + blobSasToken, blobTableName);
}

export async function prefetchEmbeddings(): Promise<void> {
  if (_prefetched !== undefined) {
    return;
  }
  _prefetched = new Map<string, Float32Array>();
  const client = getTableClient();

  const entities = client.listEntities<TableEntity>();
  for await (const entity of entities) {
    _prefetched.set(`${entity.partitionKey}|${entity.rowKey}`, new Float32Array(entity.embedding.buffer));
  }
}

export async function queryEmbedding(text: string): Promise<Float32Array> {
  prefetchEmbeddings();
  const client = getTableClient();
  const modelName = "text-embedding-ada-002";

  const embedding = new OpenAIEmbeddings({
    openAIApiKey: openAIApiKey,
    modelName: modelName
  });

  if (_prefetched && _prefetched.has(`${modelName}|${text}`)) {
    return _prefetched.get(`${modelName}|${text}`)!;
  }

  try {
    const entity = await client.getEntity<TableEntity>(modelName, text);
    return new Float32Array(entity.embedding.buffer);
  } catch (e) {
    console.log(e);
    // Errors like entity not found.
    // Fall back to OpenAI.
    const rawEmbedArray = await embedding.embedQuery(text);
    const embedArray = new Float32Array(rawEmbedArray);
    try {
      await client.createEntity<TableEntity>({
        partitionKey: modelName,
        rowKey: text,
        embedding: new Uint8Array(embedArray.buffer)
      });
    } catch (e) {
      console.error(e);
    }
    return embedArray;
  }
}
