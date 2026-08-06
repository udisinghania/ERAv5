# Wikipedia corpus-v1 review packet

This is the human validation gate between building the Wikipedia-specific quality policy and adapting quality logic to the other six data lanes.

## What to learn from this stage

The signals are measurements, the band is a policy decision derived from those measurements, the weight controls ordinary sampling, and a cap is a final safety ceiling. A cap may be configured without activating.

## Population and weighted supply

- Physical records: 5,198
- Weighted records before caps: 5,517.05
- Review examples: 50

| Band | Physical records |
|---|---:|
| B0 | 428 |
| B1 | 30 |
| B2 | 1,407 |
| B3 | 1,782 |
| B4 | 1,551 |

## Do the caps currently activate?

| Group | Records | Share after weights | Cap | Activates? |
|---|---:|---:|---:|---|
| general_short | 304 | 1.38% | 1.00% | yes |
| general_disambiguation | 106 | 0.48% | 2.00% | no |
| general_structured_low_prose | 48 | 0.22% | 2.00% | no |
| all_B0_combined | 428 | 1.94% | 5.00% | no |

Only an activating cap changes the distribution beyond the sampling weights. Non-activating caps remain useful as guards if the corpus grows later.

## Review rubric

For each example, inspect whether it contains meaningful language, is coherent and complete, belongs in its assigned band/cap group, is PII-safe, and ends cleanly. Then choose keep, downweight, or reject. Do not change a threshold after one unusual example; look for a repeated error pattern.

## Deterministic sample

Five examples are drawn across the length range of every band and cap group. All 10 non-paragraph boundary chunks are included. Some records intentionally appear in more than one stratum because they test different claims.

### band_B0

#### Squamura maculata (`rec_45a8b97da0ddb01566a0128b`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 300; paragraphs: 5; alpha: 0.813; repeated trigrams: 0.000

Beginning:

> Squamura maculata Squamura maculata is a moth in the family Cossidae. It is found on Sumatra, Borneo, Java and possibly in Cambodia. The habitat consists of lowland and lower montane forests. References Natural History Museum Lepidoptera generic names catalog Metarbelinae Moths described in 1890

Ending:

> Squamura maculata Squamura maculata is a moth in the family Cossidae. It is found on Sumatra, Borneo, Java and possibly in Cambodia. The habitat consists of lowland and lower montane forests. References Natural History Museum Lepidoptera generic names catalog Metarbelinae Moths described in 1890

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Erlau (disambiguation) (`rec_6443b2a91b1e712485aaf906`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 333; paragraphs: 4; alpha: 0.781; repeated trigrams: 0.020

Beginning:

> Erlau (disambiguation) Erlau is a municipality in the district of Mittelsachsen in Saxony in Germany. Erlau may also refer to: Erlau (Freising), a district of Freising, Bavaria, Germany Erlau (river), a river of Bavaria, Germany Erlau (Hasidic dynasty), a Haredi dynasty of Hungarian origin Eger (German: Erlau), a city in Hungary

Ending:

> Erlau (disambiguation) Erlau is a municipality in the district of Mittelsachsen in Saxony in Germany. Erlau may also refer to: Erlau (Freising), a district of Freising, Bavaria, Germany Erlau (river), a river of Bavaria, Germany Erlau (Hasidic dynasty), a Haredi dynasty of Hungarian origin Eger (German: Erlau), a city in Hungary

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Malekabad, Bazoft (`rec_061a91271010b01de8db059e`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 368; paragraphs: 4; alpha: 0.791; repeated trigrams: 0.039

Beginning:

> Malekabad, Bazoft Malekabad (, also Romanized as Mālekābād) is a village in Doab Rural District, Bazoft District, Kuhrang County, Chaharmahal and Bakhtiari Province, Iran. At the 2006 census, its population was 65, in 15 families. The village is populated by Lurs. References Populated places in Kuhrang County Luri settlements in Chaharmahal and Bakhtiari Province

Ending:

> Malekabad, Bazoft Malekabad (, also Romanized as Mālekābād) is a village in Doab Rural District, Bazoft District, Kuhrang County, Chaharmahal and Bakhtiari Province, Iran. At the 2006 census, its population was 65, in 15 families. The village is populated by Lurs. References Populated places in Kuhrang County Luri settlements in Chaharmahal and Bakhtiari Province

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### SW5 (`rec_5c7410e05b75d2407038f5fe`)

Band/weight: B0 / 0.25; caps: ['general_disambiguation']; flags: ['disambiguation_page']

Chunk: 1/1; end boundary: paragraph; characters: 498; paragraphs: 4; alpha: 0.785; repeated trigrams: 0.000

Beginning:

> SW5 SW5 may refer to: SW postcode area Earl's Court, a district in the Royal Borough of Kensington and Chelsea in central London SW5 tram, a class of electric trams, some built by the Melbourne & Metropolitan Tramways Board, but most modified from the W2 tram by the Metropolitan Transit Authority. Star Wars Episode V: The Empire Strikes Back, a 1980 American epic space opera film directed by Irvin Kershner Fernvale LRT station, Singapore See also S5W SWV (disambiguation) SW (disambiguation)

Ending:

> SW5 SW5 may refer to: SW postcode area Earl's Court, a district in the Royal Borough of Kensington and Chelsea in central London SW5 tram, a class of electric trams, some built by the Melbourne & Metropolitan Tramways Board, but most modified from the W2 tram by the Metropolitan Transit Authority. Star Wars Episode V: The Empire Strikes Back, a 1980 American epic space opera film directed by Irvin Kershner Fernvale LRT station, Singapore See also S5W SWV (disambiguation) SW (disambiguation)

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### History of baseball outside the United States (`rec_1e73f80f2cc795aca5699bd3`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 2/7; end boundary: paragraph; characters: 7,965; paragraphs: 26; alpha: 0.789; repeated trigrams: 0.037

Beginning:

> History of baseball outside the United States Baseball was open only to male amateurs in 1992 and 1996. As a result, the Americans and other nations where professional baseball is developed relied on collegiate players, while Cubans used their most experienced veterans, who technically were considered amateurs, as while they nominally held other jobs, they in fact trained full-time. In 2000, pros were admitted, but the MLB refused to release its players in 2000, 2004, and 2008, and the situation changed only a little: the Cubans still used their best players, while the Americans started using minor leaguers. The IOC cited the absence of the best players as the main reason for baseball being…

Ending:

> …Nippon Professional Baseball, consists of two leagues of 6 teams each. The country's national team has also been successful, having won two Olympic medals (bronze and silver), while the World Championships team never placed worse than 5th in its 13 appearances, winning second place once and third place three times. Recently, several Japanese players have also entered the U.S. major leagues, such as Hideo Nomo, Kazuhiro Sasaki, Ichiro Suzuki, Hideki Matsui, Kazuo Matsui, Tadahito Iguchi, Kenji Johjima, Daisuke Matsuzaka, Yu Darvish, Masahiro Tanaka, and Shohei Ohtani. Japan defeated Korea to become champions of the second World Baseball Classic on March 23, 2009, in Los Angeles. Philippines

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### band_B1

#### Caning (`rec_c4bc84475b6c1e62f93e8870`)

Band/weight: B1 / 0.5; caps: none; flags: ['short_continuation_chunk']

Chunk: 3/3; end boundary: paragraph; characters: 77; paragraphs: 2; alpha: 0.883; repeated trigrams: 0.000

Beginning:

> Caning BDSM activities Corporal punishments School punishments Whipping Pain

Ending:

> Caning BDSM activities Corporal punishments School punishments Whipping Pain

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Saint Seiya Omega (`rec_ae69d83984afd2f4d93eb5ea`)

Band/weight: B1 / 0.5; caps: none; flags: ['short_continuation_chunk']

Chunk: 2/2; end boundary: paragraph; characters: 201; paragraphs: 2; alpha: 0.836; repeated trigrams: 0.000

Beginning:

> Saint Seiya Omega Omega 2012 anime television series debuts Anime spin-offs Comics spin-offs Classical mythology in popular culture Shōnen manga TV Asahi original programming Toei Animation television

Ending:

> Saint Seiya Omega Omega 2012 anime television series debuts Anime spin-offs Comics spin-offs Classical mythology in popular culture Shōnen manga TV Asahi original programming Toei Animation television

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Gérard Condé (`rec_8e1c8aed3ca58ad319fb49c9`)

Band/weight: B1 / 0.5; caps: none; flags: ['short_continuation_chunk']

Chunk: 2/2; end boundary: paragraph; characters: 241; paragraphs: 2; alpha: 0.805; repeated trigrams: 0.156

Beginning:

> Gérard Condé 1947 births Living people Musicians from Nancy, France 20th-century French composers 21st-century French composers French male composers French music critics 20th-century French male musicians 21st-century French male musicians

Ending:

> Gérard Condé 1947 births Living people Musicians from Nancy, France 20th-century French composers 21st-century French composers French male composers French music critics 20th-century French male musicians 21st-century French male musicians

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Kenneth Leighton (`rec_2a36908894ea2b829621ce20`)

Band/weight: B1 / 0.5; caps: none; flags: ['short_continuation_chunk']

Chunk: 3/3; end boundary: paragraph; characters: 368; paragraphs: 2; alpha: 0.818; repeated trigrams: 0.061

Beginning:

> Kenneth Leighton 1929 births 1988 deaths British classical composers Alumni of The Queen's College, Oxford Academics of the University of Leeds Academics of the University of Edinburgh British classical pianists Musicians from Wakefield 20th-century classical pianists 20th-century English composers 20th-century British composers Fellows of Worcester College, Oxford

Ending:

> Kenneth Leighton 1929 births 1988 deaths British classical composers Alumni of The Queen's College, Oxford Academics of the University of Leeds Academics of the University of Edinburgh British classical pianists Musicians from Wakefield 20th-century classical pianists 20th-century English composers 20th-century British composers Fellows of Worcester College, Oxford

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### SV Arminen (`rec_34f4d41048223267e0a6d0c6`)

Band/weight: B1 / 0.5; caps: none; flags: ['low_language_content']

Chunk: 1/1; end boundary: paragraph; characters: 1,780; paragraphs: 9; alpha: 0.446; repeated trigrams: 0.161

Beginning:

> SV Arminen Sportvereinigung Arminen Wien also known as SV Arminen or simply Arminen is an Austrian professional field hockey club based in Vienna. It competes in the Austrian Bundesliga which they have won a record 19 times in the men's competition. Their home ground is the Waldstadion and they were founded on 24 April 1919. The first men's team regularly plays in the Euro Hockey League. In the past, the club also had other sports but since 1938 they only have hockey. Honours Men Austrian Bundesliga Winners (20): 1945–46, 1946–47, 1947–48, 1948–49, 1950–51, 1952–53, 1976, 1980, 1983, 1985, 1986, 1987, 1988, 2012–13, 2013–14, 2015–16, 2016–17, 2017–18, 2018–19, 2021–22 EuroHockey Club Trophy…

Ending:

> …Winners (1): 2014 Runners-up (1): 2004 Women Austrian Bundesliga Winners (20): 1948–49, 1957–58, 1978, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1992, 1998–99, 2002–03, 2011–12, 2012–13, 2013–14, 2014–15, 2015–16, 2016–17, 2017–18 Austrian Indoor Bundesliga Winners (21): 1958–59, 1980, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1992, 1997–98, 2009–10, 2010–11, 2011–12, 2013–14, 2014–15, 2015–16, 2016–17, 2017–18, 2018–19, 2019–20 EuroHockey Indoor Club Trophy Winners (3): 2011, 2015, 2019 EuroHockey Cup Winners Trophy Winners (1): 1992 References External links Field hockey clubs in Austria Field hockey clubs established in 1919 1919 establishments in Austria Sports clubs and teams in Vienna

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### band_B2

#### Pholidocarpus macrocarpus (`rec_03bdb9c44cb2ba82eeb627e8`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 400; paragraphs: 4; alpha: 0.830; repeated trigrams: 0.000

Beginning:

> Pholidocarpus macrocarpus Pholidocarpus macrocarpus is a species of flowering plant in the family Arecaceae. It is found in Peninsular Malaysia, Sumatra, and Thailand. It is threatened by habitat loss. References macrocarpus Trees of Thailand Trees of Peninsular Malaysia Trees of Sumatra Plants described in 1886 Vulnerable plants Taxa named by Odoardo Beccari Taxonomy articles created by Polbot

Ending:

> Pholidocarpus macrocarpus Pholidocarpus macrocarpus is a species of flowering plant in the family Arecaceae. It is found in Peninsular Malaysia, Sumatra, and Thailand. It is threatened by habitat loss. References macrocarpus Trees of Thailand Trees of Peninsular Malaysia Trees of Sumatra Plants described in 1886 Vulnerable plants Taxa named by Odoardo Beccari Taxonomy articles created by Polbot

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### 1981 World Orienteering Championships (`rec_bcea09d708f62ebb47ec76c5`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 539; paragraphs: 9; alpha: 0.818; repeated trigrams: 0.136

Beginning:

> 1981 World Orienteering Championships The 1981 World Orienteering Championships, the 9th World Orienteering Championships, were held in Thun, Switzerland, 3–5 September 1981. The championships had four events: individual contests for men and women, and relays for men and women. Medalists Results Men's individual Women's individual References World Orienteering Championships World Orienteering Championships International sports competitions hosted by Switzerland World Orienteering Championships Orienteering in Switzerland Thun

Ending:

> 1981 World Orienteering Championships The 1981 World Orienteering Championships, the 9th World Orienteering Championships, were held in Thun, Switzerland, 3–5 September 1981. The championships had four events: individual contests for men and women, and relays for men and women. Medalists Results Men's individual Women's individual References World Orienteering Championships World Orienteering Championships International sports competitions hosted by Switzerland World Orienteering Championships Orienteering in Switzerland Thun

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Fencing Confederation of Asia (`rec_0adcd203d204c2b0d86b7028`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 719; paragraphs: 7; alpha: 0.815; repeated trigrams: 0.062

Beginning:

> Fencing Confederation of Asia The Asian Fencing Confederation (AFC) or Fencing Confederation of Asia (FCA) is an international body created in 1988, charged with the promotion and development of fencing in Asia. It organises the Asian Fencing Championships annually, in all levels: seniors, juniors and cadet, under 23 and veterans. Colonel Houshmand Almasi, President of the Iranian Fencing Federation, was the first president of the organisation. See also Fédération Internationale d'Escrime Asian Fencing Championships References External links Asian Fencing Confederation, official site Fencing organizations Sports governing bodies in Asia 1988 establishments in Asia Sports organizations estab…

Ending:

> …eration of Asia The Asian Fencing Confederation (AFC) or Fencing Confederation of Asia (FCA) is an international body created in 1988, charged with the promotion and development of fencing in Asia. It organises the Asian Fencing Championships annually, in all levels: seniors, juniors and cadet, under 23 and veterans. Colonel Houshmand Almasi, President of the Iranian Fencing Federation, was the first president of the organisation. See also Fédération Internationale d'Escrime Asian Fencing Championships References External links Asian Fencing Confederation, official site Fencing organizations Sports governing bodies in Asia 1988 establishments in Asia Sports organizations established in 1988

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Buckshot May (`rec_d1366521feb8055f8fbd5372`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 941; paragraphs: 7; alpha: 0.749; repeated trigrams: 0.006

Beginning:

> Buckshot May William Herbert "Buckshot" May (December 13, 1899 – March 15, 1984) was a Major League Baseball pitcher who appeared in one game for the Pittsburgh Pirates in 1924. The 24-year-old right-hander stood 6'2" and weighed 169 lbs. On May 9, 1924, May came in to pitch the top of the 9th inning in a home game against the Boston Braves at Forbes Field. He pitched a scoreless inning, with one strikeout, but the Pirates lost 10–7. His lifetime ERA stands at 0.00. His manager was future Hall of Famer Bill McKechnie. Other notable teammates who would one day be members of the Baseball Hall of Fame were Max Carey, Kiki Cuyler, Rabbit Maranville, and Pie Traynor. May died in his hometown of…

Ending:

> …s. On May 9, 1924, May came in to pitch the top of the 9th inning in a home game against the Boston Braves at Forbes Field. He pitched a scoreless inning, with one strikeout, but the Pirates lost 10–7. His lifetime ERA stands at 0.00. His manager was future Hall of Famer Bill McKechnie. Other notable teammates who would one day be members of the Baseball Hall of Fame were Max Carey, Kiki Cuyler, Rabbit Maranville, and Pie Traynor. May died in his hometown of Bakersfield, California at the age of 84. External links Baseball Reference Retrosheet Major League Baseball pitchers Baseball players from Bakersfield, California Pittsburgh Pirates players Pueblo Braves players 1899 births 1984 deaths

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Anagyrus (`rec_33b86bd9804f01429f869656`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 2/3; end boundary: word; characters: 7,998; paragraphs: 2; alpha: 0.703; repeated trigrams: 0.222

Beginning:

> Anagyrus Anagyrus abatos (Noyes & Menezes, 2000) Anagyrus abdulrassouli (Myartseva, Sugonjaev & Trjapitzin, 1982) Anagyrus abyssinicus Compere, 1939 Anagyrus aceris Noyes & Hayat, 1994 Anagyrus aciculatus (Blanchard, 1940) Anagyrus adamsoni Timberlake, 1941 Anagyrus aega Noyes, 2000 Anagyrus aegyptiacus Moursi, 1948 Anagyrus agraensis Saraswat 1975 Anagyrus alami Hayat 1970 Anagyrus albatus Myartseva, 1982 Anagyrus aligarhensis Agarwal & Alam 1959 Anagyrus almoriensis Shafee, Alam & Agarwal, 1975 Anagyrus amnicus Prinsloo, 1985 Anagyrus amoenus Compere, 1939 Anagyrus amudaryensis (Myartseva, 1982) Anagyrus ananatis Gahan, 1949 Anagyrus antoninae Timberlake, 1920 Anagyrus aper Noyes & Meneze…

Ending:

> …gdianus Sugonjaev, 1968 Anagyrus sophax Noyes & Menezes 2000 Anagyrus spaici (Hoffer, 1970) Anagyrus spica (Girault 1921) Anagyrus subalbipes Ishii, 1928 Anagyrus subflaviceps (Girault 1915) Anagyrus subnigricornis Ishii, 1928 Anagyrus subproximus (Silvestri, 1915) Anagyrus subtilis Noyes & Hayat, 1994 Anagyrus sucro Noyes, 2000 Anagyrus suia Noyes, 2000 Anagyrus surekhae Noyes & Menezes 2000 Anagyrus swezeyi Timberlake, 1919 Anagyrus tamaricicola Trjapitzin, 1968 Anagyrus tanystis De Santis, 1964 Anagyrus telon Noyes & Menezes 2000 Anagyrus tenuis Noyes & Hayat, 1994 Anagyrus terebratus (Howard 1894) Anagyrus thailandicus (Myartseva, 1979) Anagyrus theana Noyes, 2000 Anagyrus theon Noyes &

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### band_B3

#### 1995 Algerian presidential election (`rec_315b8627ee867d144dfab4e2`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 1,200; paragraphs: 7; alpha: 0.798; repeated trigrams: 0.038

Beginning:

> 1995 Algerian presidential election Presidential elections were held in Algeria on 16 November 1995, in the midst of the Algerian Civil War. The result was a victory for Liamine Zeroual, head of the High Council of State at the time, who won 61% of the vote. The Armed Islamic Group of Algeria threatened to kill anyone who voted, with the slogan "one vote, one bullet", but official voter turnout was 74.9%. Candidates Liamine Zeroual, independent Mahfoud Nahnah, candidate of the Islamist Movement of Society for Peace (MSP) Said Sadi, candidate of the secularist Rally for Culture and Democracy Noureddine Boukrouh, candidate of the Party of Algerian Renewal (PRA) Conduct Delegations of observer…

Ending:

> …ment of Society for Peace (MSP) Said Sadi, candidate of the secularist Rally for Culture and Democracy Noureddine Boukrouh, candidate of the Party of Algerian Renewal (PRA) Conduct Delegations of observers came from the Arab League, the African Union, and the United Nations, and reported no major problems. The Armed Islamic Group had threatened to kill voters, but the elections passed with few attacks. Voter turnout was high, despite the three largest parties of the 1991 parliamentary elections (the Islamic Salvation Front, National Liberation Front and Socialist Forces Front) calling for a boycott. Results References Algerian Civil War Presidential elections in Algeria Presidential Algeria

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### One voice per part (`rec_5f1f77df010cc9301a40d5be`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 1,656; paragraphs: 8; alpha: 0.781; repeated trigrams: 0.018

Beginning:

> One voice per part In music, one voice per part (OVPP) is the practice of performing choral music with a single voice on each vocal line. In the specific context of Johann Sebastian Bach's works it is also known as the Rifkin hypothesis, set forth in Joshua Rifkin's 1982 article and expanded in Andrew Parrott's book The Essential Bach Choir. Choral works featuring SATB (soprano, alto, tenor and bass) vocal parts are consequently sung by four singers when this approach is adopted. The first conductor to strongly advocate this approach to the music of Bach was the American pianist and conductor Joshua Rifkin in the 1980s. The use of solo voices in the choral music of Bach has also found champ…

Ending:

> …that there are rarely additional copies of the vocal parts. Furthermore, the presence, absence and omission of solo and tutti markings in scores, as well as the ambiguity in their meaning, brings further doubt to the question of whether Bach used more than one singer per part or not. The initialism OVPP was first coined in the Internet mailing list "The Bach Recordings Discussion Group" in the mid 1990s by Steven Langley Guy. The initialism seems to have been adopted more widely since that time. References Sources Rifkin, Joshua. 1982. “Bach's Chorus: A Preliminary Report”. The Musical Times 123 (1677). Musical Times Publications Ltd.: 747–54. doi:10.2307/961592. Choral music Baroque music

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Dawn French's Boys Who Do Comedy (`rec_845a793a0368a85d7164bd39`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 2,179; paragraphs: 11; alpha: 0.801; repeated trigrams: 0.047

Beginning:

> Dawn French's Boys Who Do Comedy Dawn French's Boys Who Do Comedy is a British TV series in which comedian Dawn French interviews her favourite male comedians about how they came to be comedians. It is a follow-up and counterpart to Dawn French's Girls Who Do Comedy. The full BBC One series consists of three 30-minute programmes with 35 comedians, sewn together as if they were a single discussion. Programme 1 is about how family and early life influenced their careers, programme 2 is about the comedians' early careers, and programme 3 is about the experience of standing on stage in front of an audience. The decision was taken early in the production process to film with three cameras, large…

Ending:

> …Jackie Mason David Mitchell Paul Mooney (Episode 1 only) Graham Norton Paul O'Grady Simon Pegg Vic Reeves Paul Rodriguez (Episode 3 only) Alexei Sayle Jimmy Tarbuck David Walliams Robert Webb Paul Whitehouse Robin Williams Marc Wootton Comedians featured in clips: Episode 1: Richard Pryor, Frankie Howerd, Harpo Marx, Dudley Moore and Peter Cook, Les Dawson and Roy Barraclough, and Ade Edmondson. Episode 2: Lenny Bruce, Woody Allen, Steve Martin, Spike Milligan, and Dick Emery. Episode 3: George Carlin, Lee Evans, and Wilson, Keppel and Betty References External links 2007 British television series debuts 2007 British television series endings BBC television comedy BBC television talk shows

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Sy Barry (`rec_5c264dc79c327293bbb29d97`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 2,982; paragraphs: 11; alpha: 0.773; repeated trigrams: 0.030

Beginning:

> Sy Barry Seymour "Sy" Barry (born March 12, 1928) is an American comic-book and comic-strip artist, best known for being the artist of the strip The Phantom for more than three decades. Biography Sy Barry was born in New York City in 1928, and is the brother of comics artist Dan Barry, who drew the Flash Gordon comic strip. Sy Barry attended high school at the School of Industrial Art in Manhattan, New York City beginning in 1943. His first job as an artist was working on the comic book Famous Funnies. Barry began his professional career as his brother's art assistant, and by the late 1940s was working on his own as a freelance comic-book artist, primarily as an inker for publishers includi…

Ending:

> …equently used pencil artists on the strip, working primarily as an inker, though he often drew entire stories when time permitted. Barry's first Phantom daily strip was published on August 21, 1961 and his last on September 3, 1994. He replaced Bill Lignante on the Sundays. His first Phantom Sunday page was published on May 20, 1962 and his last on September 18, 1994. References Further reading Strickler, Dave. Syndicated Comic Strips and Artists, 1924-1995: The Complete Index. Cambria, California: Comics Access, 1995. External links Billy Ireland Cartoon Library & Museum Art Database 1928 births American comics artists Inkpot Award winners Living people High School of Art and Design alumni

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### William Weintraub (`rec_e7d38c411cbeccc16d8e3f76`)

Band/weight: B3 / 1.15; caps: none; flags: none

Chunk: 2/3; end boundary: paragraph; characters: 7,919; paragraphs: 7; alpha: 0.717; repeated trigrams: 0.159

Beginning:

> William Weintraub - documentary short, John Howe 1957 Million Dollar Smile - TV series episode, David Boisseau 1957 - writer Storm Clouds Over the Colonies - documentary short, Ronald Dick 1957 - writer Ten Days That Shook the Commonwealth - documentary short, Ronald Dick 1957 - writer Poverty and Plenty - documentary short, John Howe 1957 - writer Road to Independence - documentary short, Ronald Dick 1957 - writer They Called It White Man’s Burden - documentary short, John Howe 1957 - writer Black and White in South Africa - documentary short, John Howe & Ronald Dick 1957 - writer First Novel - documentary short, Donald Wilder 1958 - writer School for the Stage - documentary short, Julian…

Ending:

> …wards, Toronto: Genie Award for Best Theatrical Short Film, 1964 Celebration (1966) La Plata International Children's Film Festival, La Plata, Argentina: First Prize, Special Films, 1967 A Matter of Fat 22nd Canadian Film Awards, Toronto: Best Film Over 30 Minutes, 1970 Atlanta Film Festival, Atlanta: Gold Medal, 1971 Atlanta Film Festival, Atlanta: Special Jury Award, 1971 American Film and Video Festival, New York: Blue Ribbon, 1971 Jack of Hearts (1985) Chicago International Children's Film Festival, Chicago: Second Prize - Live-Action Film Under 30 Minutes, 1986 National Educational Media Network Competition, Oakland, California: Honorable Mention, Literary Adaptations, Elementary, 1986

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### band_B4

#### Pennsylvania Department of Education (`rec_c3d848e8cc61e334dfb42ada`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 4,001; paragraphs: 14; alpha: 0.830; repeated trigrams: 0.078

Beginning:

> Pennsylvania Department of Education The Pennsylvania Department of Education is the executive department of the state charged with publicly funded preschool, K-12 and adult educational budgeting, management and guidelines. As the state education agency, its activities are directed by the governor appointed Pennsylvania's Secretary of Education. The agency is headquartered at 333 Market Street in Harrisburg. The Pennsylvania Department of Education oversees 500 public school districts of Pennsylvania, over 170 public charter schools (2019), Career and Technology Centers/Vocational Technical schools, 29 Intermediate Units, the education of youth in State Juvenile Correctional Institutions, a…

Ending:

> …State Board of Education Professional Standards and Practices Commission Office of Food and Nutrition Programs Special Education Advisory Panel State Boards of Private Schools Power Library Power Library is the online portal to Pennsylvania libraries, a service of the Office of Commonwealth Libraries, Pennsylvania Department of Education. Secretaries of Education See also List of Pennsylvania state agencies State education agency References External links Official website 1837 establishments in Pennsylvania Educational administration Government agencies established in 1837 Education, Department of Department State agencies of Pennsylvania State departments of education of the United States

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Josh Lueke (`rec_5c1dc9c64b216114d8ac6502`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 5,540; paragraphs: 23; alpha: 0.783; repeated trigrams: 0.072

Beginning:

> Josh Lueke Joshua William Lueke (born December 5, 1984) is an American former professional baseball relief pitcher. He played in Major League Baseball (MLB) for the Seattle Mariners and Tampa Bay Rays, and in Nippon Professional Baseball (NPB) for the Tokyo Yakult Swallows. His impending retirement from professional baseball was announced prior to the upcoming 2022-2023 winter season of the Mexican Pacific League. Professional career Texas Rangers Lueke was drafted by the Texas Rangers in the 16th round of the 2007 Major League Baseball Draft. He played for the Spokane Indians, Clinton LumberKings, and Bakersfield Blaze. Seattle Mariners On July 9, 2010, Lueke was traded to the Seattle Mari…

Ending:

> …ines de Ciudad del Carmen players Durham Bulls players Frisco RoughRiders players Generales de Durango players Hickory Crawdads players Leones de Yucatán players Leones del Escogido players American expatriate baseball players in the Dominican Republic Long Island Ducks players Major League Baseball pitchers Mexican League baseball pitchers Nippon Professional Baseball pitchers Northern Kentucky Norse baseball players Peoria Javelinas players Pericos de Puebla players Seattle Mariners players Spokane Indians players Baseball players from Covington, Kentucky Tacoma Rainiers players Tampa Bay Rays players Tokyo Yakult Swallows players Toros del Este players West Tennessee Diamond Jaxx players

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Paul Féval, père (`rec_69bc7238cb45b0c83990605c`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/1; end boundary: paragraph; characters: 7,261; paragraphs: 25; alpha: 0.776; repeated trigrams: 0.040

Beginning:

> Paul Féval, père Paul Henri Corentin Féval, père (29 September 1816 - 8 March 1887) was a French novelist and dramatist. He was the author of popular swashbuckler novels such as Le Loup blanc (1843) and the perennial best-seller Le Bossu (1857). He also penned the seminal vampire fiction novels Le Chevalier Ténèbre (1860), La Vampire (1865) and La Ville Vampire (1874) and wrote several celebrated novels about his native Brittany and Mont Saint-Michel such as La Fée des Grèves (1850). Féval's greatest claim to fame, however, is as one of the fathers of modern crime fiction. Because of its themes and characters, his novel Jean Diable (1862) can claim to be the world's first modern novel of de…

Ending:

> …red another blow when he lost his wife. He was taken to the hospice of the Brothers of Saint-Jean de Dieu where he died on 8 March 1887. His son, Paul Féval (1860–1933) also became a prolific writer. References Sources Author and Book info.com External links Paul Féval père - Bibliographie complète at Roman-Feuilleton & HARD-BOILED site (Comprehensive Bibliographies by Vladimir Matuschenko) 1816 births 1887 deaths Writers from Rennes 19th-century French dramatists and playwrights French fantasy writers French crime fiction writers French historical novelists French horror writers University of Rennes alumni 19th-century French novelists French male novelists 19th-century French male writers

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Alachua County, Florida (`rec_9c8bc2c37a124f2487c7379f`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/3; end boundary: paragraph; characters: 7,751; paragraphs: 19; alpha: 0.786; repeated trigrams: 0.056

Beginning:

> Alachua County, Florida Alachua County ( ) is a county in the north central portion of the U.S. state of Florida. As of the 2020 census, the population was 278,468. The county seat is Gainesville, the home of the University of Florida since 1906, when the campus opened with 106 students. Alachua County is part of the Gainesville Metropolitan Statistical Area. The county is known for its diverse culture, local music, and artisans. Much of its economy revolves around the university, which had nearly 55,000 students in the fall of 2016. History Prehistory and early European settlements The first people known to have entered the area of Alachua County were Paleo-Indians, who left artifacts in t…

Ending:

> …s lynching there in 1916. These lynchings were conducted outside the justice system, by mobs or small groups working alone. Nineteen of the victims were Black; two were White. (A 2015 report by the Equal Justice Initiative, based in Montgomery, Alabama, had identified 18 lynchings. The Historical Commission documented three more, including two white men.) In September 2017, the county commission approved plans to place markers with the names of the victims in the county. (See linked article for names of these individuals.) They are working with the Historical Commission and cities to discuss how best to achieve this. A state historical marker on the Newberry Lynchings was dedicated in 2019.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Eirik Kristoffersen (`rec_cb56ee5cedef255783cbbef2`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/2; end boundary: paragraph; characters: 8,000; paragraphs: 25; alpha: 0.789; repeated trigrams: 0.059

Beginning:

> Eirik Kristoffersen Eirik Johan Kristoffersen (born 3 April 1969) is a Norwegian Army General who serves as the head of the Norwegian Armed Forces. He is a former Chief of the Norwegian Army and Norwegian Home Guard, and Chief of the Armed Forces' Special Command (FSK). Kristoffersen is the first Norwegian Chief of Defence since World War II, with battle experience. He was awarded the War Cross with Sword in 2011 for his service in Afghanistan. Military career Kristoffersen enrolled in non-commissioned officers' in 1988 and served as squad leader in the Engineer Battalion. After a few months studying engineering in college, he returned to military service in 1989 and served as squad leader…

Ending:

> …988–1989 Non-commissioned officers' training school, 1992–1995 Military Academy, Army, 2008–2009 USMC Command and Staff College, 2014–2015 United States Army War College. Authorship Kristoffersen wrote the book Jegerånden- Å lede i fred, krise og krig [ The Jäger spirit – to lead in peace, crisis and war] (2020). Awards and decorations Kristoffersen is one of Norway's most highly decorated soldiers still on active duty. He has received the following awards: Norwegian medals and awards Foreign decorations Other awards Personal life Kristoffersen lives with his wife Linn-Therece Johansen Kristoffersen and has four children from two former marriages. His interests include hunting and football.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_short

#### Squamura maculata (`rec_45a8b97da0ddb01566a0128b`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 300; paragraphs: 5; alpha: 0.813; repeated trigrams: 0.000

Beginning:

> Squamura maculata Squamura maculata is a moth in the family Cossidae. It is found on Sumatra, Borneo, Java and possibly in Cambodia. The habitat consists of lowland and lower montane forests. References Natural History Museum Lepidoptera generic names catalog Metarbelinae Moths described in 1890

Ending:

> Squamura maculata Squamura maculata is a moth in the family Cossidae. It is found on Sumatra, Borneo, Java and possibly in Cambodia. The habitat consists of lowland and lower montane forests. References Natural History Museum Lepidoptera generic names catalog Metarbelinae Moths described in 1890

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Scoparia denigata (`rec_167c1695757cf8c53e9a1e4a`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 322; paragraphs: 5; alpha: 0.758; repeated trigrams: 0.000

Beginning:

> Scoparia denigata Scoparia denigata is a moth in the family Crambidae. It was described by Harrison Gray Dyar Jr. in 1929. It has been recorded from the US state of Arizona. The wingspan is 14–18 mm. Adults are light gray brown. Adults have been recorded on wing in August. References Moths described in 1929 Scorparia

Ending:

> Scoparia denigata Scoparia denigata is a moth in the family Crambidae. It was described by Harrison Gray Dyar Jr. in 1929. It has been recorded from the US state of Arizona. The wingspan is 14–18 mm. Adults are light gray brown. Adults have been recorded on wing in August. References Moths described in 1929 Scorparia

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Schiahorn (`rec_96648c9349db8e9b6e68065c`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 349; paragraphs: 6; alpha: 0.825; repeated trigrams: 0.020

Beginning:

> Schiahorn The Schiahorn is a mountain of the Plessur Alps, overlooking Davos in the canton of Graubünden. The Schiahorn is located just east of the Strela Pass, where the summit normal route starts. References External links Schiahorn on Hikr Mountains of the Alps Mountains of Graubünden Mountains of Switzerland Two-thousanders of Switzerland

Ending:

> Schiahorn The Schiahorn is a mountain of the Plessur Alps, overlooking Davos in the canton of Graubünden. The Schiahorn is located just east of the Strela Pass, where the summit normal route starts. References External links Schiahorn on Hikr Mountains of the Alps Mountains of Graubünden Mountains of Switzerland Two-thousanders of Switzerland

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Everett Whittingham (`rec_a976f219c0f470cee2163e6b`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 373; paragraphs: 6; alpha: 0.783; repeated trigrams: 0.000

Beginning:

> Everett Whittingham Everett Whittingham (born 25 February 1954) is a Jamaican cricketer. He played in one first-class and three List A matches for the Jamaican cricket team from 1980 to 1985. See also List of Jamaican representative cricketers References External links 1954 births Living people Jamaican cricketers Jamaica cricketers Cricketers from Kingston, Jamaica

Ending:

> Everett Whittingham Everett Whittingham (born 25 February 1954) is a Jamaican cricketer. He played in one first-class and three List A matches for the Jamaican cricket team from 1980 to 1985. See also List of Jamaican representative cricketers References External links 1954 births Living people Jamaican cricketers Jamaica cricketers Cricketers from Kingston, Jamaica

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Abbas-e Aliabad (`rec_de1c9f1a6e73920fa42b73ee`)

Band/weight: B0 / 0.25; caps: ['general_short']; flags: ['short_document']

Chunk: 1/1; end boundary: paragraph; characters: 399; paragraphs: 4; alpha: 0.789; repeated trigrams: 0.049

Beginning:

> Abbas-e Aliabad Abbas-e Aliabad (, also Romanized as ʿAbbās-e ʿAlīābād) is a village in Dasht-e Zarrin Rural District, in the Central District of Kuhrang County, Chaharmahal and Bakhtiari Province, Iran. At the 2006 census, its population was 18, in 5 families. The village is populated by Lurs. References Populated places in Kuhrang County Luri settlements in Chaharmahal and Bakhtiari Province

Ending:

> Abbas-e Aliabad Abbas-e Aliabad (, also Romanized as ʿAbbās-e ʿAlīābād) is a village in Dasht-e Zarrin Rural District, in the Central District of Kuhrang County, Chaharmahal and Bakhtiari Province, Iran. At the 2006 census, its population was 18, in 5 families. The village is populated by Lurs. References Populated places in Kuhrang County Luri settlements in Chaharmahal and Bakhtiari Province

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_disambiguation

#### Robert Easton (`rec_76a398ebb1743f537331f253`)

Band/weight: B0 / 0.25; caps: ['general_short', 'general_disambiguation']; flags: ['disambiguation_page', 'short_document']

Chunk: 1/1; end boundary: paragraph; characters: 302; paragraphs: 4; alpha: 0.682; repeated trigrams: 0.000

Beginning:

> Robert Easton Robert Easton may refer to: Robert Easton (actor) (1930–2011), American actor and dialect coach Robert Easton (bass) (1898–1987), British bass singer Robert Easton (athlete) (born 1960/61), Canadian Paralympic athlete See also Robert Easton Burns (1805–1863), Canadian lawyer and judge

Ending:

> Robert Easton Robert Easton may refer to: Robert Easton (actor) (1930–2011), American actor and dialect coach Robert Easton (bass) (1898–1987), British bass singer Robert Easton (athlete) (born 1960/61), Canadian Paralympic athlete See also Robert Easton Burns (1805–1863), Canadian lawyer and judge

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Emily Martin (`rec_a27cd97f14bfcd0f5d44b7f9`)

Band/weight: B0 / 0.25; caps: ['general_short', 'general_disambiguation']; flags: ['disambiguation_page', 'short_document']

Chunk: 1/1; end boundary: paragraph; characters: 378; paragraphs: 3; alpha: 0.762; repeated trigrams: 0.000

Beginning:

> Emily Martin Emily Martin may refer to: Emily Martin (1884–1962), aka Emily Dutton, South Australian musician and socialite Emily Martin (anthropologist) (born 1944), sinologist, anthropologist, and feminist Emily Martin (rower) (born 1979), Australian rower Emily Martin (diver), British diver Emily Winfield Martin, American artist and author-illustrator of children's books

Ending:

> Emily Martin Emily Martin may refer to: Emily Martin (1884–1962), aka Emily Dutton, South Australian musician and socialite Emily Martin (anthropologist) (born 1944), sinologist, anthropologist, and feminist Emily Martin (rower) (born 1979), Australian rower Emily Martin (diver), British diver Emily Winfield Martin, American artist and author-illustrator of children's books

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### One Mississippi (`rec_e74cba94c7ccf9211a9189e2`)

Band/weight: B0 / 0.25; caps: ['general_disambiguation']; flags: ['disambiguation_page']

Chunk: 1/1; end boundary: paragraph; characters: 539; paragraphs: 3; alpha: 0.722; repeated trigrams: 0.136

Beginning:

> One Mississippi One Mississippi may refer to: One Mississippi (Brendan Benson album), 1996 One Mississippi (J Church album), 2000 One Mississippi (TV series), a 2016 American television series "One Mississippi", a song on the 2003 album Jillbilly by Jill King "One Mississippi", a song on the 2013 album Bring You Back by Brett Eldredge "One Mississippi", a song on the 2017 album So Good by Zara Larsson "One Mississippi", a song on the 2020 album My Mississippi Reunion by Steve Azar "One Mississippi" (song), a 2021 song by Kane Brown

Ending:

> One Mississippi One Mississippi may refer to: One Mississippi (Brendan Benson album), 1996 One Mississippi (J Church album), 2000 One Mississippi (TV series), a 2016 American television series "One Mississippi", a song on the 2003 album Jillbilly by Jill King "One Mississippi", a song on the 2013 album Bring You Back by Brett Eldredge "One Mississippi", a song on the 2017 album So Good by Zara Larsson "One Mississippi", a song on the 2020 album My Mississippi Reunion by Steve Azar "One Mississippi" (song), a 2021 song by Kane Brown

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Nandu (`rec_aa17c4e9fc9181141abc11d4`)

Band/weight: B0 / 0.25; caps: ['general_disambiguation']; flags: ['disambiguation_page']

Chunk: 1/1; end boundary: paragraph; characters: 830; paragraphs: 6; alpha: 0.748; repeated trigrams: 0.052

Beginning:

> Nandu Nandu may refer to: Places Chengdu, a city in Sichuan, China, known as (Southern Capital or Nandu) during the early Tang dynasty Jiangling County, a city in Hubei, China, formerly known as (Southern Capital or Nandu) during the later Tang dynasty Nandu River, Hainan province, China Other uses Ñandú, a native South American name for any of three species of Rhea. Nandu (film), a 1981 Tamil film Ñandú (vehicle), a 1940s all-terrain vehicle military vehicle Southern Metropolis Daily, often shortened to Nandu (南都) One of the Argentine Air Force flights that attacked the British fleet in the Battle of San Carlos, during the Falklands War, 1982 People with the given name Nandu Bhende (c. 195…

Ending:

> …dynasty Jiangling County, a city in Hubei, China, formerly known as (Southern Capital or Nandu) during the later Tang dynasty Nandu River, Hainan province, China Other uses Ñandú, a native South American name for any of three species of Rhea. Nandu (film), a 1981 Tamil film Ñandú (vehicle), a 1940s all-terrain vehicle military vehicle Southern Metropolis Daily, often shortened to Nandu (南都) One of the Argentine Air Force flights that attacked the British fleet in the Battle of San Carlos, during the Falklands War, 1982 People with the given name Nandu Bhende (c. 1955–2014), Indian singer Nandu M. Natekar (1933–2021), Indian badminton player See also Nandhu (born 1965), Malayalam film actor

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Old Post Office (`rec_ad1c30c210bcc53f78cd9fcd`)

Band/weight: B0 / 0.25; caps: ['general_disambiguation']; flags: ['disambiguation_page']

Chunk: 1/1; end boundary: paragraph; characters: 4,853; paragraphs: 8; alpha: 0.790; repeated trigrams: 0.342

Beginning:

> Old Post Office Old Post Office, or Former Post Office, may refer to: Serbia Old Post Office (Belgrade) United Kingdom Old Post Office, Bristol Tintagel Old Post Office, Tintagel United States (ordered by state and city) Old Athens, Alabama Main Post Office in Athens, Alabama, listed on the National Register of Historic Places Old Brick Post Office in Wickenburg, Arizona, NRHP-listed Old Camden Post Office in Camden, Arkansas, listed on the NRHP in Arkansas Old Post Office (Fayetteville, Arkansas), listed on the NRHP in Arkansas Old Post Office (Hot Springs, Arkansas), listed on the NRHP in Arkansas Little Rock U.S. Post Office and Courthouse, also known as the Old Post Office and Courthous…

Ending:

> …ria County, Texas Old Post Office (Washington, D.C.), NRHP-listed as "Old Post Office and Clock Tower" Old Post Office (Pullman, Washington), NRHP-listed as "U.S. Post Office-Pullman" Old Morgantown Post Office, part of the Monongalia Arts Center in Morgantown, West Virginia, NRHP-listed Old Ashland Post Office, listed on the NRHP in Ashland, Wisconsin Former United States Post Office (Kaukauna, Wisconsin), listed on the NRHP in Wisconsin See also Post office Postal service List of United States post offices Federal Building and Post Office (disambiguation) U.S. Post Office and Courthouse (disambiguation) Postal service (disambiguation) Post Office (disambiguation) Old Post (disambiguation)

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### cap_general_structured_low_prose

#### Futsal at the 2007 Asian Indoor Games (`rec_00ae923fa5be84e71fd2707f`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 650; paragraphs: 31; alpha: 0.754; repeated trigrams: 0.165

Beginning:

> Futsal at the 2007 Asian Indoor Games Futsal at the 2007 Asian Indoor Games was held in Macau, China from 26 October to 3 November 2007. Medalists Medal table Results Men Preliminary Group A Group B Group C Group D Kuwait was disqualified from the tournament on 29 October after Kuwait Football Association was suspended by FIFA. Knockout round Quarterfinals Semifinals Bronze medal match Gold medal match Goalscorers Women Preliminary Group A Group B Placing Knockout round Semifinals Bronze medal match Gold medal match Goalscorers References RSSSF 2007 Asian Indoor Games events Indoor Games 2007 2007 Futsal in Macau

Ending:

> Futsal at the 2007 Asian Indoor Games Futsal at the 2007 Asian Indoor Games was held in Macau, China from 26 October to 3 November 2007. Medalists Medal table Results Men Preliminary Group A Group B Group C Group D Kuwait was disqualified from the tournament on 29 October after Kuwait Football Association was suspended by FIFA. Knockout round Quarterfinals Semifinals Bronze medal match Gold medal match Goalscorers Women Preliminary Group A Group B Placing Knockout round Semifinals Bronze medal match Gold medal match Goalscorers References RSSSF 2007 Asian Indoor Games events Indoor Games 2007 2007 Futsal in Macau

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Results of the 1994 Sri Lankan general election by electoral district (`rec_c1d315b29cd48f0046377b61`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines', 'list_dominant', 'repetitive_language']

Chunk: 1/1; end boundary: paragraph; characters: 1,577; paragraphs: 11; alpha: 0.795; repeated trigrams: 0.616

Beginning:

> Results of the 1994 Sri Lankan general election by electoral district Results of the 1994 Sri Lankan general election by electoral district. Number of votes 1. The EROS/PLOTE/TELO alliance contested as TELO in Ampara District, Batticaloa District, Colombo District and Trincomalee District; as DPLF in Vanni District; and as in an independent group in Jaffna District. 2. UCPF contested as an independent group in Nuwara Eliya District. 3. EPDP contested as an independent group in Jaffna District. Percentage of votes 1. EROS contested as an independent group in four districts (Batticaloa, Jaffna, Trincomalee and Vanni). 1. The EROS/PLOTE/TELO alliance contested as TELO in Ampara District, Batti…

Ending:

> …ent group in Nuwara Eliya District. 3. EPDP contested as an independent group in Jaffna District. Seats 1. EROS contested as an independent group in four districts (Batticaloa, Jaffna, Trincomalee and Vanni). 1. The EROS/PLOTE/TELO alliance contested as TELO in Ampara District, Batticaloa District, Colombo District and Trincomalee District; as DPLF in Vanni District; and as in an independent group in Jaffna District. 2. UCPF contested as an independent group in Nuwara Eliya District. 3. EPDP contested as an independent group in Jaffna District. See also Results of the 1994 Sri Lankan general election by province References 1994 Sri Lankan parliamentary election Election results in Sri Lanka

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Coach Trip (series 8) (`rec_30d45d16b1b2d56a4fd1a10f`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 3,122; paragraphs: 10; alpha: 0.701; repeated trigrams: 0.033

Beginning:

> Coach Trip (series 8) Coach Trip 8 aired from 30 January to 9 March 2012 after the third celebrity series finished and is the eighth and final series of Coach Trip in the United Kingdom before the 2012 Summer Olympics and Paralympics. It was filmed from Monday 29 August until Saturday 1 October 2011 (after the England riots ended). The length of this series was the same as the previous non-celebrity series but with only 1 day of a weekend included at the end of the tour. The Mediterranean tour centring towards Western Asia began in the UK, before moving to the Netherlands, Germany, Austria, Italy, Greece, Bulgaria, North Macedonia (for the first time) and finishing in Turkey. Tour guide Bre…

Ending:

> …Rowing |- | 26 | Haskovo | Cooking lesson | Go-karting |- | 27 | Edirne | Soap molding | Local baths |- | 28 | Silivri | Belly dancing | Turkish wrestling |- | 29 | Istanbul | Spice market | Swimming with dolphins |- | 30 | colspan="3" |} It's the long journey home for the coach tripper and Brendan reminisces about the last six weeks on the road. References 2012 British television seasons Coach Trip series Television in North Macedonia Television shows set in Austria Television shows set in Bulgaria Television shows set in Essex Television shows set in Germany Television shows set in Greece Television shows set in Italy Television shows set in the Netherlands Television shows set in Turkey

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### 1951 Ohio State Buckeyes baseball team (`rec_8402ccaaaee1c218d8186c79`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/1; end boundary: paragraph; characters: 5,208; paragraphs: 17; alpha: 0.526; repeated trigrams: 0.281

Beginning:

> 1951 Ohio State Buckeyes baseball team The 1951 Ohio State Buckeyes baseball team represented the Ohio State University in the 1951 NCAA baseball season. The head coach was Marty Karow, serving his 1st year. The Buckeyes lost in the College World Series, defeated by the Texas A&M Aggies. Roster Schedule ! style="" | Regular season |- valign="top" |- align="center" bgcolor="#ccffcc" | 1 || March 16 || at || Unknown • San Antonio, Texas || 15–3 || 1–0 || 0–0 |- align="center" bgcolor="#ffcccc" | 2 || March 17 || at B. A. M. C. || Unknown • San Antonio, Texas || 7–8 || 1–1 || 0–0 |- align="center" bgcolor="#ffcccc" | 3 || March 19 || at || Clark Field • Austin, Texas || 0–8 || 1–2 || 0–0 |- al…

Ending:

> …mond • Columbus, Ohio || 2–4 || 22–13 || 10–2 |- align="center" bgcolor="#ccffcc" | 36 || June 9 || Western Michigan || Varsity Diamond • Columbus, Ohio || 3–2 || 23–13 || 10–2 |- |- align="center" bgcolor="#ffcccc" | 37 || June 13 || Oklahoma || Omaha Municipal Stadium • Omaha, Nebraska || 8–9 || 23–14 || 10–2 |- align="center" bgcolor="#ffcccc" | 38 || June 13 || Texas A&M || Omaha Municipal Stadium • Omaha, Nebraska || 2–3 || 23–15 || 10–2 |- Awards and honors Dick Hauck First Team All-Big Ten Stewart Hein First Team All-Big Ten References Ohio State Buckeyes baseball seasons Ohio State Buckeyes baseball Big Ten Conference baseball champion seasons Ohio State College World Series seasons

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### History of baseball outside the United States (`rec_1e73f80f2cc795aca5699bd3`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 2/7; end boundary: paragraph; characters: 7,965; paragraphs: 26; alpha: 0.789; repeated trigrams: 0.037

Beginning:

> History of baseball outside the United States Baseball was open only to male amateurs in 1992 and 1996. As a result, the Americans and other nations where professional baseball is developed relied on collegiate players, while Cubans used their most experienced veterans, who technically were considered amateurs, as while they nominally held other jobs, they in fact trained full-time. In 2000, pros were admitted, but the MLB refused to release its players in 2000, 2004, and 2008, and the situation changed only a little: the Cubans still used their best players, while the Americans started using minor leaguers. The IOC cited the absence of the best players as the main reason for baseball being…

Ending:

> …Nippon Professional Baseball, consists of two leagues of 6 teams each. The country's national team has also been successful, having won two Olympic medals (bronze and silver), while the World Championships team never placed worse than 5th in its 13 appearances, winning second place once and third place three times. Recently, several Japanese players have also entered the U.S. major leagues, such as Hideo Nomo, Kazuhiro Sasaki, Ichiro Suzuki, Hideki Matsui, Kazuo Matsui, Tadahito Iguchi, Kenji Johjima, Daisuke Matsuzaka, Yu Darvish, Masahiro Tanaka, and Shohei Ohtani. Japan defeated Korea to become champions of the second World Baseball Classic on March 23, 2009, in Los Angeles. Philippines

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

### boundary_non_paragraph

#### List of Saxifragales of South Africa (`rec_032e77599758be3cf3243d37`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 3/4; end boundary: sentence; characters: 7,974; paragraphs: 2; alpha: 0.812; repeated trigrams: 0.345

Beginning:

> List of Saxifragales of South Africa leucantha (Schonland & Baker f.) Toelken, endemic Crassula multiflora Schonland & Baker f. subsp. multiflora, endemic Crassula muricata Thunb. endemic Crassula muscosa L. indigenous Crassula muscosa L. var. muscosa, indigenous Crassula muscosa L. var. obtusifolia (Harv.) G.D.Rowley, indigenous Crassula muscosa L. var. parvula (Eckl. & Zeyh.) Toelken, endemic Crassula muscosa L. var. polpodacea (Eckl. & Zeyh.) G.D.Rowley, endemic Crassula namaquensis Schonland & Baker f. indigenous Crassula namaquensis Schonland & Baker f. subsp. comptonii (Hutch. & Pillans) Toelken, endemic Crassula namaquensis Schonland & Baker f. subsp. lutea (Schonland) Toelken, endem…

Ending:

> …, endemic Crassula thunbergiana Schult. indigenous Crassula thunbergiana Schult. subsp. minutiflora (Schonland & Baker f.) Toelken, indigenous Crassula thunbergiana Schult. subsp. thunbergiana, endemic Crassula tomentosa Thunb. indigenous Crassula tomentosa Thunb. var. glabrifolia (Harv.) G.D.Rowley, indigenous Crassula tomentosa Thunb. var. tomentosa, indigenous Crassula tuberella Toelken, indigenous Crassula umbella Jacq. endemic Crassula umbellata Thunb. endemic Crassula umbraticola N.E.Br. indigenous Crassula vaginata Eckl. & Zeyh. indigenous Crassula vaginata Eckl. & Zeyh. subsp. vaginata, indigenous Crassula vaillantii (Willd.) Roth, not indigenous, naturalised Crassula vestita Thunb.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Solanales of South Africa (`rec_1e8861f09d317fec99bc2f92`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['repetitive_language']

Chunk: 3/4; end boundary: sentence; characters: 7,938; paragraphs: 7; alpha: 0.832; repeated trigrams: 0.435

Beginning:

> List of Solanales of South Africa Lycium Genus Lycium: Lycium acutifolium E.Mey. ex Dunal, endemic Lycium afrum L. endemic Lycium amoenum Dammer, indigenous Lycium arenicola Miers, indigenous Lycium bosciifolium Schinz, indigenous Lycium cinereum Thunb. indigenous Lycium cordatum Mill. accepted as Carissa bispinosa (L.) Desf. ex Brenan, indigenous Lycium ferocissimum Miers, indigenous Lycium gariepense A.M.Venter, indigenous Lycium grandicalyx Joubert & Venter, indigenous Lycium hantamense A.M.Venter, indigenous Lycium hirsutum Dunal, indigenous Lycium horridum Thunb. indigenous Lycium mascarenense A.M.Venter & A.J.Scott, indigenous Lycium oxycarpum Dunal, endemic Lycium pilifolium C.H.Wrig…

Ending:

> …uivi Lam. indigenous Solanum jasminoides Paxton, accepted as Solanum laxum Spreng. not indigenous, naturalised Solanum kibweziense Dammer, accepted as Solanum tettense Klotzsch Solanum koniortodes Dammer, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright, accepted as Solanum tettense Klotzsch, indigenous Solanum kwebense N.E.Br. ex C.H.Wright var. acutius Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. chondropetalum (Dammer) Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var. luederitzii (Schinz) Bitter, accepted as Solanum tettense Klotzsch Solanum kwebense N.E.Br. ex C.H.Wright var.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### First-wave feminism (`rec_37928ac5a43c3bb20dfd038c`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 9/11; end boundary: sentence; characters: 7,948; paragraphs: 2; alpha: 0.800; repeated trigrams: 0.178

Beginning:

> First-wave feminism 1896 Argentina: A group of anarcha-feminist women, headed by Virginia Bolten, publish La Voz de la Mujer, one of the first feminist newspapers of Latin America. US, Idaho: Idaho granted women the right to vote. 1900 Western Australia: Western Australia granted women the right to vote. Belgium: Legal majority was granted to unmarried women. Egypt: A school for female teachers was founded in Cairo. France: Women were allowed to practice law. Korea: The post office profession was opened to women. Tunisia: The first public elementary school for girls was opened. Japan: The first women's university was opened. Baden, Germany: Universities opened to women. Sweden: Maternity le…

Ending:

> …ast London Federation of Suffragettes. 1913 Russia: In 1913 Russian women observed their first International Women's Day on the last Sunday in February. Following discussions, International Women's Day was transferred to 8 March and this day has remained the global date for International Women's Day ever since. US, Alaska: Alaska granted women the right to vote. Norway: Norway granted women the right to vote. Japan: Public universities opened to women. United Kingdom: The suffragette Emily Davison was killed by the King's horse at The Derby. United Kingdom: 50,000 women taking part in a pilgrimage organized by the National Union of Women's Suffrage Societies arrived in Hyde Park on July 26.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### First-wave feminism (`rec_aab9ffe0e7274b82d98aa5f4`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 8/11; end boundary: sentence; characters: 7,909; paragraphs: 2; alpha: 0.778; repeated trigrams: 0.167

Beginning:

> First-wave feminism 1839 US, Mississippi: Mississippi was the first U.S. state that gave married women limited property rights. United Kingdom: The Custody of Infants Act 1839 made it possible for divorced mothers to be granted custody of their children under seven, but only if the Lord Chancellor agreed to it, and only if the mother was of good character. US, Mississippi: The Married Women's Property Act 1839 granted married women the right to own (but not control) property in their own name. 1840 US, Texas: Married women were allowed to own property in their own name. 1841 Bulgaria: The first secular girls school in Bulgaria was opened, making education and the profession of teacher avail…

Ending:

> …n the right to vote. New Zealand: New Zealand became the first self-governing country in the world in which all women had the right to vote in parliamentary elections. Cook Islands: The Cook Islands granted women the right to vote in island councils and a federal parliament. 1894 South Australia: South Australia granted women the right to vote. United Kingdom: The United Kingdom extended the right to vote in local elections to married women. 1895 US: Almost all U.S. states had passed some form of Sole Trader Laws, Property Laws, and Earnings Laws, granting married women the right to trade without their husbands' consent, own and/or control their own property, and control their own earnings.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Saxifragales of South Africa (`rec_afcd42c6126b6516c4786d22`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 2/4; end boundary: sentence; characters: 7,957; paragraphs: 2; alpha: 0.816; repeated trigrams: 0.348

Beginning:

> List of Saxifragales of South Africa parvisepala (Schonland) Toelken, indigenous Crassula alcicornis Schonland, endemic Crassula alpestris Thunb. indigenous Crassula alpestris Thunb. subsp. alpestris, endemic Crassula alpestris Thunb. subsp. massonii (Britten & Baker f.) Toelken, endemic Crassula alstonii Marloth, endemic Crassula ammophila Toelken, endemic Crassula aphylla Schonland & Baker f. endemic Crassula arborea Medik. accepted as Crassula arborescens (Mill.) Willd. subsp. arborescens, indigenous Crassula arborescens (Mill.) Willd. endemic Crassula arborescens (Mill.) Willd. subsp. arborescens, endemic Crassula arborescens (Mill.) Willd. subsp. undulatifolia Toelken, endemic Crassula…

Ending:

> ….) D.Dietr. indigenous Crassula mesembryanthemoides (Haw.) D.Dietr. subsp. hispida (Haw.) Toelken, endemic Crassula mesembryanthemoides (Haw.) D.Dietr. subsp. mesembryanthemoides, endemic Crassula minuta Toelken, endemic Crassula mollis Thunb. endemic Crassula montana Thunb. indigenous Crassula montana Thunb. subsp. montana, endemic Crassula montana Thunb. subsp. quadrangularis (Schonland) Toelken, endemic Crassula multicava Lem. indigenous Crassula multicava Lem. subsp. floribunda Friedrich ex Toelken, endemic Crassula multicava Lem. subsp. multicava, endemic Crassula multiceps Harv. endemic Crassula multiflora Schonland & Baker f. indigenous Crassula multiflora Schonland & Baker f. subsp.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of moths of Australia (Cosmopterigidae) (`rec_ba813b8299138077ae565e30`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 1/2; end boundary: sentence; characters: 6,703; paragraphs: 3; alpha: 0.716; repeated trigrams: 0.154

Beginning:

> List of moths of Australia (Cosmopterigidae) This is a list of the Australian species of the family Cosmopterigidae. It also acts as an index to the species articles and forms part of the full List of moths of Australia. Chrysopeleiinae Cholotis exodroma (Meyrick, 1897) Cholotis semnostola (Meyrick, 1897) Eumenodora encrypta Meyrick, 1906 Ithome lassula Hodges, 1962 Leptozestis anagrapta (Meyrick, 1897) Leptozestis antithetis (Meyrick, 1897) Leptozestis argoscia (Lower, 1904) Leptozestis autochroa (Meyrick, 1915) Leptozestis capnopora (Meyrick, 1897) Leptozestis cataspoda (Meyrick, 1897) Leptozestis charmosyna (Meyrick, 1921) Leptozestis crassipalpis (Turner, 1923) Leptozestis crebra (Meyri…

Ending:

> …yrick, 1920 Limnaecia cirrhosema Turner, 1923 Limnaecia cirrhozona Turner, 1923 Limnaecia crossomela Lower, 1908 Limnaecia cybophora Meyrick, 1897 Limnaecia definitiva (T.P. Lucas, 1901) Limnaecia elaphropa Turner, 1923 Limnaecia epimictis Meyrick, 1897 Limnaecia eristica Meyrick, 1919 Limnaecia eugramma Lower, 1899 Limnaecia hemidoma Meyrick, 1897 Limnaecia hemimitra Turner, 1923 Limnaecia heterozona Lower, 1904 Limnaecia ida Lower, 1908 Limnaecia iriastis Meyrick, 1897 Limnaecia isodesma Lower, 1904 Limnaecia isozona Meyrick, 1897 Limnaecia leptomeris Meyrick, 1897 Limnaecia leptozona Turner, 1923 Limnaecia leucomita Turner, 1923 Limnaecia loxoscia Lower, 1923 Limnaecia lunacrescens (T.P.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### William Weintraub (`rec_e5a81bf7f25eef4a19c4a329`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/3; end boundary: sentence; characters: 4,988; paragraphs: 15; alpha: 0.771; repeated trigrams: 0.056

Beginning:

> William Weintraub William Weintraub (February 19, 1926 – November 6, 2017) was a Canadian documentarian/filmmaker, journalist and author, best known for his long career with the National Film Board of Canada (NFB). Early life Weintraub was born in Montreal, to Louis Weintraub and Mina Blumer Weintraub, and grew up in the blue-collar neighbourhood of Verdun. His father had been a stock broker; he lost everything in the Wall Street Crash of 1929 and worked as the manager of a corner store. William studied English Literature and political science at McGill University, where he had worked on the McGill Daily. In 1947, he took the job of a ski reporter at The Montreal Gazette, from which he was…

Ending:

> …r Wing – documentary short, Don Haldane 1956 - writer Saskatchewan Traveller - documentary short, Don Haldane 1956 - writer Portrait of the Family - documentary short, Ronald Dick 1957 - writer The Invisible Keystone - documentary short, Ronald Dick 1957 - writer Four Centuries of Growing Pains - documentary short, Ronald Dick, Nicholas Balla 1957 - writer The Colonies Look Ahead - documentary short, Ronald Dick 1957 - writer Can It Hold Together? - documentary short, Ronald Dick 1957 Crisis in Asia - documentary short, Ronald Dick 1957 The Ghost That Talked - documentary short, Don Haldane 1957 - writer A Letter from Oxford - documentary short, Julian Biggs 1957 Colonialism: Ogre or Angel?

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### FC Luch Vladivostok (`rec_f643d420641475d418d95ff1`)

Band/weight: B0 / 0.25; caps: ['general_structured_low_prose']; flags: ['duplicate_lines']

Chunk: 1/2; end boundary: sentence; characters: 7,581; paragraphs: 11; alpha: 0.688; repeated trigrams: 0.349

Beginning:

> FC Luch Vladivostok FC Luch Vladivostok () was an association football club based in Vladivostok, Russia. In 2005, Luch won the Russian First Division and played in the Premier League from 2006 to 2008. The club was called Luch-Energiya from 2003 to 2018, when it was renamed due to sponsorship from Dalenergo, an energy distribution company. History Luch has been playing in the Soviet Union championship since 1958. The name Luch means Ray. The club played in the Far East regional tournament of "B-class" teams and eventually won it in 1965, earning promotion to "A-class". Luch played in this regional tournament until league reorganization in 1972. From 1972 to 1991, Luch played in the Eastern…

Ending:

> …gn=center|30 |align=center|12 |align=center|5 |align=center|13 |align=center|37 |align=center|39 |align=center|41 |align=center|R16 |align=center colspan="2"|— |align=left| A. Ivanov – 5 |align=left| Pavlov |- |align=center|2007 |align=center|14 |align=center|30 |align=center|8 |align=center|8 |align=center|14 |align=center|26 |align=center|38 |align=center|32 |align=center|R32 |align=center colspan="2"|— |align=left| Strelkov – 5 |align=left| Pavlov |- |align=center|2008 |align=center bgcolor="pink"|16 |align=center|30 |align=center|3 |align=center|12 |align=center|15 |align=center|24 |align=center|53 |align=center|21 |align=center|R32 |align=center colspan="2"|— |align=left| Bulyga – 5 I.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### List of Saxifragales of South Africa (`rec_fb334bc7112873c9f00ca154`)

Band/weight: B4 / 1.25; caps: none; flags: none

Chunk: 1/4; end boundary: sentence; characters: 7,931; paragraphs: 15; alpha: 0.807; repeated trigrams: 0.169

Beginning:

> List of Saxifragales of South Africa Saxifragales (saxifrages) is an order of flowering plants (Angiosperms). They are an extremely diverse group of plants which include trees, shrubs, perennial herbs, succulent and aquatic plants. The degree of diversity in terms of vegetative and floral features makes it difficult to define common features that unify the order. In the Angiosperm Phylogeny Group classification system, the Saxifragales are placed within the major division of flowering plants referred to as eudicots, specifically the core eudicots. This subgroup consists of the Dilleniaceae, superasterids and superrosids. The superrosids in turn have two components, rosids and Saxifragales.…

Ending:

> …Druce, indigenous Cotyledon papillaris L.f. indigenous Cotyledon pendens Van Jaarsv. endemic Cotyledon petiolaris Van Jaarsv. endemic Cotyledon rhombifolia Haw. accepted as Adromischus rhombifolius (Haw.) Lem. Cotyledon tomentosa Harv. indigenous Cotyledon tomentosa Harv. subsp. ladismithiensis (Poelln.) Toelken, endemic Cotyledon tomentosa Harv. subsp. tomentosa, endemic Cotyledon velutina Hook.f. endemic Cotyledon woodii Schonland & Baker f. endemic Cotyledon xanthantha Van Jaarsv. & Eggli, endemic Crassula Genus Crassula: Crassula acinaciformis Schinz, indigenous Crassula alba Forssk. var. alba, indigenous Crassula alba Forssk. var. pallida Toelken, indigenous Crassula alba Forssk. var.

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:

#### Anagyrus (`rec_33b86bd9804f01429f869656`)

Band/weight: B2 / 1.0; caps: none; flags: none

Chunk: 2/3; end boundary: word; characters: 7,998; paragraphs: 2; alpha: 0.703; repeated trigrams: 0.222

Beginning:

> Anagyrus Anagyrus abatos (Noyes & Menezes, 2000) Anagyrus abdulrassouli (Myartseva, Sugonjaev & Trjapitzin, 1982) Anagyrus abyssinicus Compere, 1939 Anagyrus aceris Noyes & Hayat, 1994 Anagyrus aciculatus (Blanchard, 1940) Anagyrus adamsoni Timberlake, 1941 Anagyrus aega Noyes, 2000 Anagyrus aegyptiacus Moursi, 1948 Anagyrus agraensis Saraswat 1975 Anagyrus alami Hayat 1970 Anagyrus albatus Myartseva, 1982 Anagyrus aligarhensis Agarwal & Alam 1959 Anagyrus almoriensis Shafee, Alam & Agarwal, 1975 Anagyrus amnicus Prinsloo, 1985 Anagyrus amoenus Compere, 1939 Anagyrus amudaryensis (Myartseva, 1982) Anagyrus ananatis Gahan, 1949 Anagyrus antoninae Timberlake, 1920 Anagyrus aper Noyes & Meneze…

Ending:

> …gdianus Sugonjaev, 1968 Anagyrus sophax Noyes & Menezes 2000 Anagyrus spaici (Hoffer, 1970) Anagyrus spica (Girault 1921) Anagyrus subalbipes Ishii, 1928 Anagyrus subflaviceps (Girault 1915) Anagyrus subnigricornis Ishii, 1928 Anagyrus subproximus (Silvestri, 1915) Anagyrus subtilis Noyes & Hayat, 1994 Anagyrus sucro Noyes, 2000 Anagyrus suia Noyes, 2000 Anagyrus surekhae Noyes & Menezes 2000 Anagyrus swezeyi Timberlake, 1919 Anagyrus tamaricicola Trjapitzin, 1968 Anagyrus tanystis De Santis, 1964 Anagyrus telon Noyes & Menezes 2000 Anagyrus tenuis Noyes & Hayat, 1994 Anagyrus terebratus (Howard 1894) Anagyrus thailandicus (Myartseva, 1979) Anagyrus theana Noyes, 2000 Anagyrus theon Noyes &

Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]

Decision: keep [ ]  downweight [ ]  reject [ ]

Notes:
