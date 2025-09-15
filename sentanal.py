"""Legacy runner maintained for backward compatibility.

This script now delegates to the new, modular training pipeline in
``twitter_sentiment``. Run `python -m twitter_sentiment.train --help` for options.
"""

from twitter_sentiment.train import main

if __name__ == '__main__':
    main()
