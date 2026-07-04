import argparse

from parse_text import DPModel2Parser


def main():
    parser = argparse.ArgumentParser(description="DPModel2 parsing test.")
    parser.add_argument("--checkpoint_path", type=str,default="/rydata/jinliye/LanTracking/DPModel2/ckpt/best_model_epoch_273_uas_0.8729_las_0.8521.pth")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_spacy", type=int, default=1)
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--sentence", type=str, default="")
    parser.add_argument("--sent_file", type=str, default="")
    args = parser.parse_args()

    parser_model = DPModel2Parser(
        checkpoint_path=args.checkpoint_path,
        cache_dir=args.cache_dir,
        device=args.device,
        use_spacy=bool(args.use_spacy),
        spacy_model=args.spacy_model,
    )

    sentences = []
    if args.sent_file:
        with open(args.sent_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
    elif args.sentence:
        sentences = [args.sentence]
    else:
        sentences = [
            "a white dog running in the road",
            "the red car near the building",
            "a man riding a bicycle in the park",
        ]

    for s in sentences:
        out = parser_model.extract_triplet(s)
        print(f"Sentence: {s}")
        print(f"Target: {out['target']}")
        print(f"Concepts: {out['concepts']}")
        print(f"Background: {out['background']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
