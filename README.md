[![DOI](https://zenodo.org/badge/442209987.svg)](https://zenodo.org/badge/latestdoi/442209987)
# TED-S: Twitter Event Dataset with Sentiments

TED-S is a Twitter dataset corresponding to two major events (from sports and political domain) throughout a continuous 
period with both sub-event and sentiment labels.

TED-S is an extended version of [Twitter-Event-Data-2019](https://github.com/HHansi/Twitter-Event-Data-2019) (TED), and sub-event ground truth data are available in the TED repository.

## Events
1. MUNLIV - English Premier League 19/20 match between Manchester United FC and Liverpool FC on October 20, 2019
2. BrexitVote - Brexit Super Saturday 2019/ UK parliament session on Saturday, October 19, 2019

| Event       | Period (UTC)| Total Tweets | Non-duplicates |
| ----------- | ----------: | -----------: | -------------: |
| MUNLIV      | 15:28-16:23 | 99,837       | 41,721         |
| BrexitVote  | 08:00-13:59 | 174,078      | 35,541         |

More details covering the annotation approaches and sentiment distributions are available in the paper "[TED-S: Twitter Event Data in Sports and Politics with Aggregated Sentiments](https://www.mdpi.com/2306-5729/7/7/90)".

### Reference
```
@Article{data7070090,
  AUTHOR = {Hettiarachchi, Hansi and Al-Turkey, Doaa and Adedoyin-Olowe, Mariam and Bhogal, Jagdev and Gaber, Mohamed Medhat},
  TITLE = {{TED-S}: Twitter Event Data in Sports and Politics with Aggregated Sentiments},
  JOURNAL = {Data},
  VOLUME = {7},
  YEAR = {2022},
  NUMBER = {7},
  ARTICLE-NUMBER = {90},
  URL = {https://www.mdpi.com/2306-5729/7/7/90},
  ISSN = {2306-5729},
  DOI = {10.3390/data7070090}
}
```