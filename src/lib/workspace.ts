import type { Dataset } from './dataset';
import type { FtArtifact } from './xformer/ft';
import type { TrainedHead } from './trainedHead';

export interface WorkspaceItems {
  dataset: Dataset;
  trainedHead: TrainedHead;
  ftArtifact: FtArtifact;
}

const DB_NAME = 'anvil-workspace';
const STORE_NAME = 'items';
let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  if (typeof indexedDB === 'undefined') {
    return Promise.reject(new Error('Local workspace storage is not available in this browser.'));
  }
  const pending = new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE_NAME, { keyPath: 'key' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Could not open local workspace storage.'));
    request.onblocked = () => reject(new Error('Local workspace storage upgrade is blocked.'));
  }).catch((error: unknown) => {
    dbPromise = null;
    throw error;
  });
  dbPromise = pending;
  return pending;
}

export async function getWorkspaceItem<K extends keyof WorkspaceItems>(
  key: K,
): Promise<WorkspaceItems[K] | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(key);
    request.onsuccess = () => resolve((request.result?.value as WorkspaceItems[K] | undefined) ?? null);
    request.onerror = () => reject(request.error ?? new Error('Could not read local workspace data.'));
  });
}

export async function setWorkspaceItem<K extends keyof WorkspaceItems>(
  key: K,
  value: WorkspaceItems[K] | null,
): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    if (value === null) store.delete(key);
    else store.put({ key, value });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('Could not save local workspace data.'));
    tx.onabort = () => reject(tx.error ?? new Error('Local workspace save was aborted.'));
  });
}
