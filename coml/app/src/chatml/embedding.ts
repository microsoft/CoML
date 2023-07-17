import { odata, TableClient, AzureNamedKeyCredential } from "@azure/data-tables";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { blobAccountName, blobAccountKey, blobTableName, blobUrl, openAIApiKey } from "./apiKey";

async function testEmbedding() {
  const creds = new AzureNamedKeyCredential(blobAccountName, blobAccountKey);
  const client = new TableClient(blobUrl, blobTableName, creds);
  const modelName = "text-embedding-ada-002";

  const embedding = new OpenAIEmbeddings({
    openAIApiKey: openAIApiKey,
    modelName: modelName
  });

  const message = "Hello world!";
  const array = await embedding.embedQuery(message);
  console.log(array);
  console.log(Buffer.from(array));

  await client.createEntity({
    partitionKey: modelName,
    rowKey: message,
    embedding: Buffer.from(array)
  })

}

testEmbedding();