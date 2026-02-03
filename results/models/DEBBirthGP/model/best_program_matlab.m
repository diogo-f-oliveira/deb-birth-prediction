function y = best_program_matlab(g, k, f, v_Hb)
%GP_BEST_PROGRAM  MATLAB translation of the evolved GP expression.
%
% Inputs can be scalars or arrays of the same size (elementwise ops used).
% g, k, f, v_Hb must be real; v_Hb should be >= 0 for sqrt.

    % Helper: nested atan(atan(...atan(g))) eight times
    a1 = atan(g);
    a2 = atan(a1);
    a3 = atan(a2);
    a4 = atan(a3);
    a5 = atan(a4);
    a6 = atan(a5);
    a7 = atan(a6);
    a8 = atan(a7);   % this equals atan(atan(... eight times ... atan(g)))

    sv = sqrt(v_Hb);

    % Common term: (4*f*atan(atan(...)) + sqrt(v_Hb)*(2*k + 1)^2)
    A = 4.*f.*a8 + sv.*(2.*k + 1).^2;

    % ---- Build the nested Min/Max parts exactly as in the expression ----

    % X = (4*f*(g + k) + sqrt(v_Hb)*(2*g + 1)*(2*k + 1))/(4*f)
    X = (4.*f.*(g + k) + sv.*(2.*g + 1).*(2.*k + 1)) ./ (4.*f);

    % inner = Min(X, sqrt(v_Hb)*(2*k*Max(k^3, sqrt(v_Hb)) + 1)/(2*f))
    inner = min(X, sv .* (2.*k.*max(k.^3, sv) + 1) ./ (2.*f));

    % M = Max(sqrt(v_Hb), inner)
    M = max(sv, inner);

    % Y = sqrt(v_Hb)*(2*k*M + 1)/(2*f)
    Y = sv .* (2.*k.*M + 1) ./ (2.*f);

    % B = Min(X, Y)
    B = min(X, Y);

    % C = Min( (k*sqrt(v_Hb)*(4*k + 1) + 2*atan(g))/(sqrt(v_Hb)*(4*k + 1)) , 2*atan(g) + 3 )
    C1 = (k.*sv.*(4.*k + 1) + 2.*atan(g)) ./ (sv.*(4.*k + 1));
    C2 = 2.*atan(g) + 3;
    C  = min(C1, C2);

    % Full expression:
    % (12*f^6 - v_Hb^(3/2)*A*B*C) / (f^5 * A)
    y = (12.*f.^6 - (v_Hb.^(3/2)).*A.*B.*C) ./ (f.^5 .* A);
end
