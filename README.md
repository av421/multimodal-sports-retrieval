# Multimodal Sports Retrieval

Search a sports action image dataset by **uploading a photo** or **typing a description** (e.g. "diving into a pool", "swinging a golf club") and get back the most visually/semantically similar images — powered by CLIP embeddings and FAISS nearest-neighbor search.

> Status: work in progress. This README will be filled in with architecture, setup instructions, real evaluation numbers, and a demo GIF as the project is built.

## Roadmap

- [x] Data pipeline (Kaggle "100 Sports Image Classification" dataset)
- [ ] CLIP embedding pipeline (open_clip, ViT-B/32)
- [ ] FAISS index over image embeddings
- [ ] Unified text/image search function
- [ ] Evaluation (precision@k, recall@k, confusion breakdown)
- [ ] Gradio demo app
- [ ] Final README polish (architecture diagram, results, screenshots, limitations)

## License

MIT — see [LICENSE](LICENSE).
