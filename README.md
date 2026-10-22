# DigitalHealthCareTeam1

## 작업 흐름
SNUHub 서버는 외부 인터넷이 차단되어 있고, GitHub는 캐시 서버를 통해 **읽기만** 가능합니다.

1. 로컬 PC에서 코드 수정 → `git push`
2. 서버에서 `git pull` 후 실행
3. 서버에서는 코드를 직접 수정하지 않기 (pull 충돌 방지, 서버에서 push 불가)

서버 최초 clone:
```bash
cd ~/workspace
git clone --recurse-submodules http://github.snuhub-cache-server/Gyeong-Rak/DigitalHealthCareTeam1.git
```

이미 clone한 경우 / 서브모듈 갱신:
```bash
git pull && git submodule update --init --recursive
```

`EVA-X/`는 [hustvl/EVA-X](https://github.com/hustvl/EVA-X) 서브모듈입니다. URL을 상대 경로(`../../hustvl/EVA-X.git`)로 지정해 두어서 로컬에서는 github.com, 서버에서는 캐시 서버에서 받아집니다.

## 주의
- **Public 저장소입니다.** 환자 데이터, 결과 파일, 계정/경로 정보, API 키를 올리지 마세요.
- 데이터·가중치·결과물은 서버에만 두고 `.gitignore`로 제외합니다.
