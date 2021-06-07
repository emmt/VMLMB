%% subfunction that checks if we are in Octave.
function r = is_octave()
    persistent x;
    if isempty(x)
        x = (exist("OCTAVE_VERSION", "builtin") ~= 0);
    end
    r = x;
end
