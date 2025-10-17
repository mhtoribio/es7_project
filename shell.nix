{ pkgs ? import <nixpkgs> { } }:
let
  py = pkgs.python312;
  pythonEnv = py.withPackages (ps: [ ps.numpy ]);
in pkgs.mkShell {
  buildInputs = [ pythonEnv ];
  shellHook = ''
    if [ ! -d .venv ]; then
      ${py.interpreter} -m venv .venv
    fi
    . .venv/bin/activate
    export PYTHONPATH="${pythonEnv}/${py.sitePackages}:$PYTHONPATH"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PYTHONNOUSERSITE=1
  '';
}
