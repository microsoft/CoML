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

function canonicalizeRowKey(rowKey: string): string {
  return rowKey
    .replaceAll("/", "%")  // https://stackoverflow.com/questions/47047107/azure-storage-size-of-partition-key-out-of-range
    .replaceAll("\\", "%")
    .replaceAll("#", "%")
    .replaceAll("?", "%")
    .replaceAll("\t", "%")
    .replaceAll("\n", "%")
    .replaceAll("\r", "%")
    .slice(0, 960);  // Key can't be too long
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

export async function preprocessEmbeddings(texts: string[]): Promise<void> {
  await prefetchEmbeddings();
  const client = getTableClient();
  const modelName = "text-embedding-ada-002";

  const embedding = new OpenAIEmbeddings({
    openAIApiKey: openAIApiKey,
    modelName: modelName
  });

  const filteredTexts = texts.filter((text) => !_prefetched!.has(`${modelName}|${canonicalizeRowKey(text)}`));
  console.log(texts.length, filteredTexts.length);
  const batchSize = 32;
  for (let index = 0; index < filteredTexts.length; index += batchSize) {
    const rawEmbedArrays = await embedding.embedDocuments(filteredTexts.slice(index, index + batchSize));
    const embedArrays = rawEmbedArrays.map((rawEmbedArray) => new Float32Array(rawEmbedArray));
    for (let i = 0; i < embedArrays.length; i++) {
      try {
        await client.createEntity<TableEntity>({
          partitionKey: modelName,
          rowKey: canonicalizeRowKey(filteredTexts[index + i]),
          embedding: new Uint8Array(embedArrays[i].buffer)
        });
      } catch (e) {
        console.error(e);
      }
    }
  }
}

export async function queryEmbedding(text: string): Promise<Float32Array> {
  await prefetchEmbeddings();
  const client = getTableClient();
  const modelName = "text-embedding-ada-002";

  const embedding = new OpenAIEmbeddings({
    openAIApiKey: openAIApiKey,
    modelName: modelName
  });

  if (_prefetched && _prefetched.has(`${modelName}|${canonicalizeRowKey(text)}`)) {
    return _prefetched.get(`${modelName}|${canonicalizeRowKey(text)}`)!;
  }

  try {
    const entity = await client.getEntity<TableEntity>(modelName, text);
    const embedArray = new Float32Array(entity.embedding.buffer);
    if (_prefetched) {
      _prefetched.set(`${modelName}|${canonicalizeRowKey(text)}`, embedArray);
    }
    return embedArray;
  } catch (e) {
    console.log(e);
    // Errors like entity not found.
    // Fall back to OpenAI.
    const rawEmbedArray = await embedding.embedQuery(text);
    const embedArray = new Float32Array(rawEmbedArray);
    if (_prefetched) {
      _prefetched.set(`${modelName}|${canonicalizeRowKey(text)}`, embedArray);
    }
    try {
      await client.createEntity<TableEntity>({
        partitionKey: modelName,
        rowKey: canonicalizeRowKey(text),
        embedding: new Uint8Array(embedArray.buffer)
      });
    } catch (e) {
      console.error(e);
    }
    return embedArray;
  }
}
