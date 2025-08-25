// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <msgpack.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    double ts;
    void **events;
    size_t events_count;
} KVEventBatch;

typedef struct
{
    int type; // 0: BlockStored, 1: BlockRemoved, 2: AllBlocksCleared
    union
    {
        struct
        {
            int64_t *block_hashes;
            size_t block_hashes_count;
            int64_t parent_block_hash;
            int64_t *token_ids;
            size_t token_ids_count;
            int32_t block_size;
            int64_t lora_id;
        } block_stored;
        struct
        {
            int64_t *block_hashes;
            size_t block_hashes_count;
        } block_removed;
    } data;
} KVCacheEvent;

static int parse_block_stored(msgpack_object_map *map, KVCacheEvent *event)
{
    event->type = 0;

    for (size_t i = 0; i < map->size; i++)
    {
        msgpack_object_kv *kv = &map->ptr[i];

        if (kv->key.type == MSGPACK_OBJECT_STR &&
            kv->key.via.str.size == 12 &&
            memcmp(kv->key.via.str.ptr, "block_hashes", 12) == 0)
        {
            if (kv->val.type == MSGPACK_OBJECT_ARRAY)
            {
                event->data.block_stored.block_hashes_count = kv->val.via.array.size;
                event->data.block_stored.block_hashes = malloc(kv->val.via.array.size * sizeof(int64_t));
                for (size_t j = 0; j < kv->val.via.array.size; j++)
                {
                    if (kv->val.via.array.ptr[j].type == MSGPACK_OBJECT_POSITIVE_INTEGER)
                    {
                        event->data.block_stored.block_hashes[j] = kv->val.via.array.ptr[j].via.u64;
                    }
                }
            }
        }
        else if (kv->key.type == MSGPACK_OBJECT_STR &&
                 kv->key.via.str.size == 17 &&
                 memcmp(kv->key.via.str.ptr, "parent_block_hash", 17) == 0)
        {
            if (kv->val.type == MSGPACK_OBJECT_POSITIVE_INTEGER)
            {
                event->data.block_stored.parent_block_hash = kv->val.via.u64;
            }
            else if (kv->val.type == MSGPACK_OBJECT_NIL)
            {
                event->data.block_stored.parent_block_hash = -1;
            }
        }
        else if (kv->key.type == MSGPACK_OBJECT_STR &&
                 kv->key.via.str.size == 9 &&
                 memcmp(kv->key.via.str.ptr, "token_ids", 9) == 0)
        {
            if (kv->val.type == MSGPACK_OBJECT_ARRAY)
            {
                event->data.block_stored.token_ids_count = kv->val.via.array.size;
                event->data.block_stored.token_ids = malloc(kv->val.via.array.size * sizeof(int64_t));
                for (size_t j = 0; j < kv->val.via.array.size; j++)
                {
                    if (kv->val.via.array.ptr[j].type == MSGPACK_OBJECT_POSITIVE_INTEGER)
                    {
                        event->data.block_stored.token_ids[j] = kv->val.via.array.ptr[j].via.u64;
                    }
                }
            }
        }
        else if (kv->key.type == MSGPACK_OBJECT_STR &&
                 kv->key.via.str.size == 10 &&
                 memcmp(kv->key.via.str.ptr, "block_size", 10) == 0)
        {
            if (kv->val.type == MSGPACK_OBJECT_POSITIVE_INTEGER)
            {
                event->data.block_stored.block_size = kv->val.via.u64;
            }
        }
        else if (kv->key.type == MSGPACK_OBJECT_STR &&
                 kv->key.via.str.size == 7 &&
                 memcmp(kv->key.via.str.ptr, "lora_id", 7) == 0)
        {
            if (kv->val.type == MSGPACK_OBJECT_POSITIVE_INTEGER)
            {
                event->data.block_stored.lora_id = kv->val.via.u64;
            }
            else if (kv->val.type == MSGPACK_OBJECT_NIL)
            {
                event->data.block_stored.lora_id = -1;
            }
        }
    }
    return 0;
}

static int parse_block_removed(msgpack_object_map *map, KVCacheEvent *event)
{
    event->type = 1;

    for (size_t i = 0; i < map->size; i++)
    {
        msgpack_object_kv *kv = &map->ptr[i];

        if (kv->key.type == MSGPACK_OBJECT_STR &&
            kv->key.via.str.size == 12 &&
            memcmp(kv->key.via.str.ptr, "block_hashes", 12) == 0)
        {
            if (kv->val.type == MSGPACK_OBJECT_ARRAY)
            {
                event->data.block_removed.block_hashes_count = kv->val.via.array.size;
                event->data.block_removed.block_hashes = malloc(kv->val.via.array.size * sizeof(int64_t));
                for (size_t j = 0; j < kv->val.via.array.size; j++)
                {
                    if (kv->val.via.array.ptr[j].type == MSGPACK_OBJECT_POSITIVE_INTEGER)
                    {
                        event->data.block_removed.block_hashes[j] = kv->val.via.array.ptr[j].via.u64;
                    }
                }
            }
        }
    }
    return 0;
}

static int parse_all_blocks_cleared(msgpack_object_map *map, KVCacheEvent *event)
{
    event->type = 2;
    return 0;
}

static int parse_kv_event(msgpack_object *obj, KVCacheEvent *event)
{
    if (obj->type != MSGPACK_OBJECT_MAP)
    {
        return -1;
    }

    msgpack_object_map *map = &obj->via.map;
    const char *event_type = NULL;

    for (size_t i = 0; i < map->size; i++)
    {
        msgpack_object_kv *kv = &map->ptr[i];

        if (kv->key.type == MSGPACK_OBJECT_STR &&
            kv->key.via.str.size == 4 &&
            memcmp(kv->key.via.str.ptr, "type", 4) == 0)
        {
            if (kv->val.type == MSGPACK_OBJECT_STR)
            {
                event_type = kv->val.via.str.ptr;
                break;
            }
        }
    }

    if (!event_type)
    {
        return -1;
    }

    if (strncmp(event_type, "BlockStored", 11) == 0)
    {
        return parse_block_stored(map, event);
    }
    else if (strncmp(event_type, "BlockRemoved", 12) == 0)
    {
        return parse_block_removed(map, event);
    }
    else if (strncmp(event_type, "AllBlocksCleared", 16) == 0)
    {
        return parse_all_blocks_cleared(map, event);
    }

    return -1;
}

KVEventBatch *parse_kv_event_batch(const void *payload, size_t length)
{
    msgpack_unpacked result;
    msgpack_unpacked_init(&result);

    if (msgpack_unpack_next(&result, (const char *)payload, length, NULL) != MSGPACK_UNPACK_SUCCESS)
    {
        msgpack_unpacked_destroy(&result);
        return NULL;
    }

    msgpack_object *obj = &result.data;
    if (obj->type != MSGPACK_OBJECT_MAP)
    {
        msgpack_unpacked_destroy(&result);
        return NULL;
    }

    KVEventBatch *batch = malloc(sizeof(KVEventBatch));
    memset(batch, 0, sizeof(KVEventBatch));

    msgpack_object_map *map = &obj->via.map;
    for (size_t i = 0; i < map->size; i++)
    {
        msgpack_object_kv *kv = &map->ptr[i];

        if (kv->key.type == MSGPACK_OBJECT_STR &&
            kv->key.via.str.size == 2 &&
            memcmp(kv->key.via.str.ptr, "ts", 2) == 0)
        {
            if (kv->val.type == MSGPACK_OBJECT_FLOAT)
            {
                batch->ts = kv->val.via.f64;
            }
        }
        else if (kv->key.type == MSGPACK_OBJECT_STR &&
                 kv->key.via.str.size == 6 &&
                 memcmp(kv->key.via.str.ptr, "events", 6) == 0)
        {
            if (kv->val.type == MSGPACK_OBJECT_ARRAY)
            {
                batch->events_count = kv->val.via.array.size;
                batch->events = malloc(kv->val.via.array.size * sizeof(KVCacheEvent *));

                for (size_t j = 0; j < kv->val.via.array.size; j++)
                {
                    KVCacheEvent *event = malloc(sizeof(KVCacheEvent));
                    memset(event, 0, sizeof(KVCacheEvent));

                    if (parse_kv_event(&kv->val.via.array.ptr[j], event) == 0)
                    {
                        batch->events[j] = event;
                    }
                    else
                    {
                        free(event);
                        batch->events[j] = NULL;
                    }
                }
            }
        }
    }

    msgpack_unpacked_destroy(&result);
    return batch;
}

void free_kv_event_batch(KVEventBatch *batch)
{
    if (!batch)
        return;

    for (size_t i = 0; i < batch->events_count; i++)
    {
        if (batch->events[i])
        {
            KVCacheEvent *event = batch->events[i];
            if (event->type == 0)
            {
                free(event->data.block_stored.block_hashes);
                free(event->data.block_stored.token_ids);
            }
            else if (event->type == 1)
            {
                free(event->data.block_removed.block_hashes);
            }
            free(event);
        }
    }

    free(batch->events);
    free(batch);
}

void print_kv_event_batch(const KVEventBatch *batch)
{
    printf("Timestamp: %.3f\n", batch->ts);
    printf("Events count: %zu\n", batch->events_count);

    for (size_t i = 0; i < batch->events_count; i++)
    {
        const KVCacheEvent *event = batch->events[i];
        if (!event)
            continue;

        printf("Event %zu: ", i);
        switch (event->type)
        {
        case 0:
            printf("BlockStored\n");
            printf("  Block hashes: ");
            for (size_t j = 0; j < event->data.block_stored.block_hashes_count; j++)
            {
                printf("%ld ", event->data.block_stored.block_hashes[j]);
            }
            printf("\n  Parent block hash: %ld\n", event->data.block_stored.parent_block_hash);
            printf("  Token IDs: ");
            for (size_t j = 0; j < event->data.block_stored.token_ids_count; j++)
            {
                printf("%ld ", event->data.block_stored.token_ids[j]);
            }
            printf("\n  Block size: %d\n", event->data.block_stored.block_size);
            printf("  Lora ID: %ld\n", event->data.block_stored.lora_id);
            break;
        case 1:
            printf("BlockRemoved\n");
            printf("  Block hashes: ");
            for (size_t j = 0; j < event->data.block_removed.block_hashes_count; j++)
            {
                printf("%ld ", event->data.block_removed.block_hashes[j]);
            }
            printf("\n");
            break;
        case 2:
            printf("AllBlocksCleared\n");
            break;
        }
    }
}